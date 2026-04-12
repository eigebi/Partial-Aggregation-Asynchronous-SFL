from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ExperimentCfg
from models import ClientModelBank, Keysets, State
from batch_stream import ClientBatchStream
from async_env import ClientCaps, EdgeCaps, AsyncSystemEnv, _clip_pos, _mbps_to_bps, _bytes_to_bits
from system_profile import ensure_profile_bank_from_cfg

Tensor = torch.Tensor


@dataclass
class ClientRuntimeState:
    base_version: int = 0
    prefix_version: int = 0
    progress: float = 0.0
    virtual_queue: float = 0.0


class SplitFedEngine:
    """
    Minimal engine aligned with cfg.asyncenv.scheme:
        0 = sync
        1 = FedAsync  -> fllike aggregation
        2 = FedBuffer -> fllike aggregation (typically min_ready_clients > 1)
        3 = Proposed  -> dual aggregation

    Design principles:
      - run-specific exogenous inputs stay explicit:
          client_splits / client_caps / edge_caps
      - device / profile_bank / env are built inside __init__ from cfg
      - event timing and physical delay use async_env
      - dual update semantics follow the original implementation:
          suffix: sum(all advanced clients' suffix deltas) / k
          prefix: sum(ready clients' prefix deltas vs anchor) / num_ready
    """

    def __init__(
        self,
        cfg: ExperimentCfg,
        bank: ClientModelBank,
        train_set,
        test_set,
        client_splits: List[List[int]],
        client_caps: List[ClientCaps],
        edge_caps: EdgeCaps,
        initial_cut_idx: Optional[int] = None,
        runtime_seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.bank = bank
        self.train_set = train_set
        self.test_set = test_set

        # explicit run-specific inputs
        self.client_splits = client_splits
        self.client_caps = client_caps
        self.edge_caps = edge_caps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_clients = int(cfg.asyncenv.num_clients)
        if self.num_clients != len(client_splits):
            raise ValueError("len(client_splits) must equal cfg.asyncenv.num_clients")
        if self.num_clients != len(client_caps):
            raise ValueError("len(client_caps) must equal cfg.asyncenv.num_clients")
        if self.num_clients != bank.num_clients:
            raise ValueError("bank.num_clients must equal cfg.asyncenv.num_clients")

        self.scheme = int(cfg.asyncenv.scheme)
        self.H = int(cfg.asyncenv.client_cotrain_steps)
        self.k = int(cfg.asyncenv.min_ready_clients)
        self.cut_idx = int(cfg.asyncenv.init_cut_idx if initial_cut_idx is None else initial_cut_idx)
        self.exp_gamma = float(cfg.asyncenv.exp_gamma)
        self.runtime_seed = int(cfg.seed if runtime_seed is None else runtime_seed)

        self.profile_bank = ensure_profile_bank_from_cfg(cfg, device=self.device.type, force_regen=False)
        self.env = AsyncSystemEnv(clients=client_caps, edge=edge_caps, profile_bank=self.profile_bank)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.bank.model.parameters(),
            lr=float(cfg.train.lr),
            momentum=float(getattr(cfg.train, "momentum", 0.0)),
            weight_decay=float(getattr(cfg.train, "weight_decay", 0.0)),
            foreach=True,
        )

        self.server_version = 0
        self.global_phy_time = 0.0
        self.client_state: Dict[int, ClientRuntimeState] = {
            cid: ClientRuntimeState() for cid in range(self.num_clients)
        }

        self.keysets: Optional[Keysets] = None
        if self.scheme == 3:
            self.keysets = self.bank.get_keysets(self.cut_idx)
            self.bank.ensure_prefix_anchor(self.keysets)

        self.batch_streams: Dict[int, ClientBatchStream] = {}
        seed_base = self.runtime_seed + 10000
        for cid in range(self.num_clients):
            self.batch_streams[cid] = ClientBatchStream(
                dataset=self.train_set,
                indices=self.client_splits[cid],
                batch_size=int(cfg.train.batch_size),
                seed=seed_base + cid,
                device=self.device,
            )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=int(cfg.train.test_batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg.train, "num_workers", 2)),
            pin_memory=(self.device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # config / state helpers
    # ------------------------------------------------------------------

    def set_cut_idx(self, cut_idx: int) -> None:
        self.cut_idx = int(cut_idx)
        if self.scheme == 3:
            self.keysets = self.bank.get_keysets(self.cut_idx)
            self.bank.ensure_prefix_anchor(self.keysets)

    def _default_edge_alloc(self) -> np.ndarray:
        return np.ones(self.num_clients, dtype=np.float64)

    def _cut_vector(self) -> np.ndarray:
        return np.full(self.num_clients, self.cut_idx, dtype=np.int32)

    def _copy_nonfloat_from_first(self, ids: List[int]) -> None:
        if not ids:
            return
        first = self.bank.client_base_state[ids[0]]
        with torch.no_grad():
            for k in self.bank.nonfloat_keys:
                self.bank.server_state[k].copy_(first[k])

    # ------------------------------------------------------------------
    # split timing helpers (same formulas as async_env step 1+2)
    # ------------------------------------------------------------------

    def _split_step_time_vector(
        self,
        client_cut_layers: np.ndarray,
        edge_alloc_norm: Optional[np.ndarray] = None,
        mul_client_fb: float = 3.0,
        mul_edge_fb: float = 3.0,
    ) -> np.ndarray:
        N = self.num_clients
        cut_vec = np.asarray(client_cut_layers, dtype=np.int32)
        if cut_vec.shape != (N,):
            raise ValueError(f"client_cut_layers must have shape ({N},)")

        if edge_alloc_norm is None:
            a = np.ones(N, dtype=np.float64)
        else:
            a = np.asarray(edge_alloc_norm, dtype=np.float64)
        s = float(np.maximum(a, 0.0).sum())
        if s <= 0.0:
            a = np.ones(N, dtype=np.float64) / float(N)
        else:
            a = np.maximum(a, 0.0) / s

        edge_flops_i = _clip_pos(a * float(self.edge_caps.total_flops))
        t_step = np.zeros(N, dtype=np.float64)

        c_flops_list = [c.flops for c in self.client_caps]
        c_ul_bps_list = [_mbps_to_bps(c.ul_mbps) for c in self.client_caps]
        c_dl_bps_list = [_mbps_to_bps(c.dl_mbps) for c in self.client_caps]

        for i in range(N):
            prof = self.profile_bank.get(int(cut_vec[i]))
            flops_pre = float(prof.flops_prefix)
            flops_suf = float(prof.flops_suffix)
            bits_smash = _bytes_to_bits(float(prof.bytes_smash_ul))
            bits_grad = _bytes_to_bits(float(prof.bytes_smashgrad_dl))

            t_client_comp = float(mul_client_fb) * flops_pre / _clip_pos(np.array([c_flops_list[i]]))[0]
            t_edge_comp = float(mul_edge_fb) * flops_suf / edge_flops_i[i]
            t_ul = bits_smash / _clip_pos(np.array([c_ul_bps_list[i]]))[0]
            t_dl = bits_grad / _clip_pos(np.array([c_dl_bps_list[i]]))[0]
            t_step[i] = t_client_comp + t_edge_comp + t_ul + t_dl

        return t_step

    # ------------------------------------------------------------------
    # batch / train / eval
    # ------------------------------------------------------------------

    def _next_batch(self, cid: int) -> Tuple[Tensor, Tensor]:
        x, y = self.batch_streams[cid].next_batch()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    def _local_train_from_base(self, cid: int, base_state: State, nsteps: int) -> None:
        if nsteps <= 0:
            return
        self.bank.copy_state_to_model(base_state)
        self.bank.model.train()

        for _ in range(int(nsteps)):
            x, y = self._next_batch(cid)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.bank.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

    def _train_client_steps(self, cid: int, nsteps: int) -> None:
        if nsteps <= 0:
            return
        base_state = self.bank.client_base_state[cid]
        self._local_train_from_base(cid, base_state, nsteps)
        with torch.no_grad():
            self.bank.copy_model_to_state(base_state)

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        self.bank.copy_state_to_model(self.bank.server_state)
        self.bank.model.eval()

        total = 0
        correct = 0
        loss_sum = 0.0
        for x, y in self.test_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.bank.model(x)
            loss = self.criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())

        self.bank.model.train()
        return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)

    # ------------------------------------------------------------------
    # staleness helpers
    # ------------------------------------------------------------------

    def _current_staleness_vector(self) -> np.ndarray:
        """
        scheme 0/1/2:
            staleness_i = server_version - base_version_i

        scheme 3:
            staleness_i = server_version - prefix_version_i
            because suffix is broadcast to all every event, while prefix
            is only synced for ready clients.
        """
        if self.scheme == 3:
            st = np.array(
                [self.server_version - self.client_state[cid].prefix_version for cid in range(self.num_clients)],
                dtype=np.float32,
            )
        else:
            st = np.array(
                [self.server_version - self.client_state[cid].base_version for cid in range(self.num_clients)],
                dtype=np.float32,
            )

        if np.any(st < 0):
            raise RuntimeError("Negative staleness detected, which should not happen.")
        return st

    # ------------------------------------------------------------------
    # event progress / delay
    # ------------------------------------------------------------------

    def _advance_to_k_complete_and_get_steps(self, Hf: float, k: int, eps: float = 1e-12):
        H = int(Hf)
        N = self.num_clients

        p0 = np.array([self.client_state[cid].progress for cid in range(N)], dtype=np.float64)
        step_time = self._split_step_time_vector(self._cut_vector(), self._default_edge_alloc())
        speeds = 1.0 / np.maximum(step_time, eps)

        done_mask = p0 >= (Hf - eps)
        num_done = int(np.sum(done_mask))

        if num_done >= k:
            dt = 0.0
        else:
            need = k - num_done
            rem_time = np.full(N, np.inf, dtype=np.float64)
            active_mask = ~done_mask & (speeds > 0)
            rem_time[active_mask] = (Hf - p0[active_mask]) / speeds[active_mask]
            valid_rems = rem_time[~done_mask]
            if len(valid_rems) < need:
                dt = 0.0
            else:
                dt = float(np.partition(valid_rems, need - 1)[need - 1])
                if not np.isfinite(dt) or dt < 0:
                    dt = 0.0

        p1 = np.clip(p0 + speeds * dt, a_min=None, a_max=Hf)
        s0 = np.floor(p0 + eps).astype(int)
        s1 = np.floor(p1 + eps).astype(int)
        delta = np.maximum(s1 - s0, 0)
        cap_to_H = np.maximum(H - s0, 0)
        steps_evt = np.minimum(delta, cap_to_H)

        ready = [cid for cid in range(N) if p1[cid] >= (Hf - eps)]
        for cid in ready:
            steps_evt[cid] = cap_to_H[cid]

        for cid in range(N):
            self.client_state[cid].progress = float(p1[cid])

        return dt, ready, steps_evt.tolist(), p1.tolist()

    def _event_delay(self, ready: List[int], steps_evt: List[int]) -> float:
        if len(ready) == 0:
            return 0.0
        cuts = np.full(self.num_clients, self.cut_idx, dtype=np.int32)
        H_vec = np.array(steps_evt, dtype=np.float64)
        res = self.env.event_latency(
            client_cut_layers=cuts,
            edge_alloc_norm=np.ones(self.num_clients, dtype=np.float64),
            H_vec=H_vec,
            ready_vec=ready,
        )
        return float(res.get("T_event", 0.0))

    # ------------------------------------------------------------------
    # aggregation helpers
    # ------------------------------------------------------------------

    def _simple_average_into_server(self, ids: List[int], float_keys: List[str]) -> None:
        if len(ids) == 0:
            return
        src_states = [self.bank.client_base_state[cid] for cid in ids]
        weights = [1.0] * len(ids)
        self.bank.assign_weighted_average_into_state(
            dst=self.bank.server_state,
            src_states=src_states,
            weights=weights,
            float_keys=float_keys,
            normalize=True,
        )
        self._copy_nonfloat_from_first(ids)

    def _fedasync_hinge_coeff(self, stale: int) -> float:
        """
        alpha_t = alpha * hinge(stale)

        hinge(u) = 1,                    if u <= b
                = 1 / (a * (u - b) + 1), if u > b

        Minimal choice: reuse cfg.asyncenv.exp_gamma as base alpha.
        Use the paper-style hinge defaults a=10, b=4.
        """
        alpha = float(self.exp_gamma)
        a = 5.0
        b = 10

        u = max(0, int(stale))
        if u <= b:
            scale = 1.0
        else:
            scale = 1.0 / (a * float(u - b) + 1.0)

        mix = alpha * scale
        return float(np.clip(mix, 0.0, 1.0))


    def _mix_single_client_into_server(self, cid: int, mix: float) -> None:
        """
        server <- (1-mix) * server + mix * local_client_model
        """
        local_state = self.bank.client_base_state[cid]

        with torch.no_grad():
            for kk in self.bank.float_keys:
                self.bank.server_state[kk].mul_(1.0 - mix).add_(mix * local_state[kk])

        self._copy_nonfloat_from_first([cid])

    def _fedasync_mix_coeff(self, stale: int) -> float:
        """
        Minimal FedAsync-style staleness scaling:
            alpha_t = alpha / (stale + 1)

        We reuse cfg.asyncenv.exp_gamma as the base alpha.
        """
        alpha = float(self.exp_gamma)
        mix = alpha / float(max(0, stale) + 1)
        return float(np.clip(mix, 0.0, 1.0))


    def _mix_single_client_into_server(self, cid: int, mix: float) -> None:
        """
        server <- (1-mix) * server + mix * local_client_model
        """
        local_state = self.bank.client_base_state[cid]

        with torch.no_grad():
            for kk in self.bank.float_keys:
                self.bank.server_state[kk].mul_(1.0 - mix).add_(mix * local_state[kk])

        self._copy_nonfloat_from_first([cid])

    # ------------------------------------------------------------------
    # scheme 0: sync
    # ------------------------------------------------------------------

    def _run_sync_event(self) -> int:
        ids = list(range(self.num_clients))
        steps_evt = [self.H for _ in range(self.num_clients)]

        for cid in ids:
            self._train_client_steps(cid, self.H)

        self._simple_average_into_server(ids, self.bank.float_keys)
        self.server_version += 1

        for cid in ids:
            self.client_state[cid].progress = 0.0
            self.client_state[cid].base_version = self.server_version
            self.client_state[cid].prefix_version = self.server_version
            self.bank.copy_state_to_state(self.bank.client_base_state[cid], self.bank.server_state)

        self.global_phy_time += self._event_delay(ids, steps_evt)
        return len(ids)

    def _run_async_event_fedasync_hinge(self) -> int:
        """
        scheme 1: FedAsync-hinge
        - trigger when at least one client becomes ready
        - process one arrived client at a time
        - server update uses hinge-scaled interpolation
        """
        Hf = float(self.H)

        # FedAsync is one-arrival-triggered
        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, 1)

        # local training for ALL advanced clients
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue

            local_state = self.bank.client_base_state[cid]
            self._local_train_from_base(cid, local_state, nsteps)

            with torch.no_grad():
                for kk in self.bank.float_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])
                for kk in self.bank.nonfloat_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])

        if ready:
            # If several clients finish at exactly the same event time,
            # consume one deterministically; the rest will be handled by later zero-time events.
            accepted_cid = min(ready)

            stale = self.server_version - self.client_state[accepted_cid].base_version
            mix = self._fedasync_hinge_coeff(stale)

            self._mix_single_client_into_server(accepted_cid, mix)
            self.server_version += 1

            self.client_state[accepted_cid].progress = 0.0
            self.client_state[accepted_cid].base_version = self.server_version
            self.bank.copy_state_to_state(
                self.bank.client_base_state[accepted_cid],
                self.bank.server_state,
            )

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)


    def _run_async_event_fedasync(self) -> int:
        """
        scheme 1: FedAsync-style
        - trigger when at least one client becomes ready
        - server updates with ONE arrived client at a time
        - update is staleness-scaled interpolation, not direct overwrite
        """
        Hf = float(self.H)

        # FedAsync should be one-arrival-triggered
        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, 1)

        # local training for ALL advanced clients
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue

            local_state = self.bank.client_base_state[cid]
            self._local_train_from_base(cid, local_state, nsteps)

            with torch.no_grad():
                for kk in self.bank.float_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])
                for kk in self.bank.nonfloat_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])

        if ready:
            # if multiple clients complete at exactly the same event time,
            # process one deterministically; the others remain ready and will
            # be consumed by subsequent zero-time events
            accepted_cid = min(ready)

            stale = self.server_version - self.client_state[accepted_cid].base_version
            mix = self._fedasync_mix_coeff(stale)

            self._mix_single_client_into_server(accepted_cid, mix)
            self.server_version += 1

            self.client_state[accepted_cid].progress = 0.0
            self.client_state[accepted_cid].base_version = self.server_version
            self.client_state[accepted_cid].prefix_version = self.server_version
            self.bank.copy_state_to_state(
                self.bank.client_base_state[accepted_cid],
                self.bank.server_state,
            )

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)
    # ------------------------------------------------------------------
    # scheme 2: async_fllike
    # ------------------------------------------------------------------

    def _run_async_event_fllike(self) -> int:
        Hf = float(self.H)
        k = max(1, min(int(self.k), self.num_clients))

        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, k)

        # local training for ALL advanced clients
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue

            local_state = self.bank.client_base_state[cid]
            self._local_train_from_base(cid, local_state, nsteps)

            with torch.no_grad():
                for kk in self.bank.float_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])
                for kk in self.bank.nonfloat_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])

        accepted = list(ready)

        # fllike = directly average READY clients' current local models
        if accepted:
            self._simple_average_into_server(accepted, self.bank.float_keys)
            self.server_version += 1

            for cid in accepted:
                self.client_state[cid].progress = 0.0
                self.client_state[cid].base_version = self.server_version
                self.client_state[cid].prefix_version = self.server_version
                self.bank.copy_state_to_state(
                    self.bank.client_base_state[cid],
                    self.bank.server_state,
                )

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)


    def _run_async_event_delta_fedbuff(self) -> int:
        """
        scheme 4: DeltaFedBuff (correct cycle-anchor version)

        - Every client accumulates local training toward a fixed budget H.
        - Once progress reaches H, it stops training (progress capped).
        - Trigger an event when >=k clients have completed H.
        - Aggregation uses ONLY ready clients.
        - Each ready client uploads delta vs its cycle anchor (full model float keys).
        - Server applies the averaged delta to CURRENT server.
        """
        Hf = float(self.H)
        if Hf <= 0:
            raise ValueError("client_cotrain_steps must be positive")

        k = max(1, min(int(self.k), self.num_clients))

        # IMPORTANT: full anchor is cycle anchor, not event anchor
        self.bank.ensure_full_anchor()

        # 1) shared physics
        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, k)

        # 2) local training accumulation for ALL advanced clients
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue

            local_state = self.bank.client_base_state[cid]
            self._local_train_from_base(cid, local_state, nsteps)

            # write back FULL local state so unfinished clients continue next event
            with torch.no_grad():
                for kk in self.bank.float_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])
                for kk in self.bank.nonfloat_keys:
                    local_state[kk].copy_(self.bank.tensor_ref[kk])

        # 3) aggregation: ready clients upload delta vs their CYCLE anchor
        tmp_acc = self.bank.tmp_delta
        assert tmp_acc is not None

        with torch.no_grad():
            for kk in self.bank.float_keys:
                tmp_acc[kk].zero_()

            accepted = list(ready)
            for cid in accepted:
                anchor = self.bank.client_full_anchor_state[cid]
                local_state = self.bank.client_base_state[cid]
                for kk in self.bank.float_keys:
                    tmp_acc[kk].add_(local_state[kk] - anchor[kk])

            # 4) server update (averaged delta apply)
            if accepted:
                inv = 1.0 / float(len(accepted))
                for kk in self.bank.float_keys:
                    self.bank.server_state[kk].add_(inv * tmp_acc[kk])
                self.server_version += 1

        # 5) broadcast + reset ONLY accepted; next cycle starts
        for cid in accepted:
            self.client_state[cid].progress = 0.0
            self.client_state[cid].base_version = self.server_version

            # broadcast float keys to local model
            self.bank.copy_state_to_state(
                self.bank.client_base_state[cid],
                self.bank.server_state,
                is_float_keys=True,
            )

            # refresh THIS client's cycle anchor to current server
            self.bank.copy_server_float_into_full_anchor(cid)

        # 6) time
        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(accepted)
    # ------------------------------------------------------------------
    # scheme 3: async_dual
    # ------------------------------------------------------------------

    def _run_async_event_dual(self) -> int:
        Hf = float(self.H)
        k = max(1, min(int(self.k), self.num_clients))

        ks = self.keysets
        if ks is None:
            self.keysets = self.bank.get_keysets(self.cut_idx)
            self.bank.ensure_prefix_anchor(self.keysets)
            ks = self.keysets
        assert ks is not None

        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, k)
        ready_set = set(ready)

        acc_pre = self.bank.tmp_base
        acc_suf = self.bank.tmp_delta
        assert acc_pre is not None and acc_suf is not None

        with torch.no_grad():
            for kk in ks.client_float_keys:
                acc_pre[kk].zero_()
            for kk in ks.server_float_keys:
                acc_suf[kk].zero_()

        num_ready = 0
        num_advanced = 0

        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue
            num_advanced += 1

            base_state = self.bank.client_base_state[cid]
            anchor_state = self.bank.client_prefix_anchor_state[cid]
            self._local_train_from_base(cid, base_state, nsteps)

            with torch.no_grad():
                # keep local non-float / buffers
                for kk in self.bank.nonfloat_keys:
                    base_state[kk].copy_(self.bank.tensor_ref[kk])

                # suffix: sum ALL advanced contributions, divide by k later
                for kk in ks.server_float_keys:
                    acc_suf[kk].add_(self.bank.tensor_ref[kk] - base_state[kk])

                # prefix: READY only, delta contribution to fed server
                if cid in ready_set:
                    for kk in ks.client_float_keys:
                        acc_pre[kk].add_(self.bank.tensor_ref[kk] - anchor_state[kk])
                    num_ready += 1
                else:
                    # unfinished clients keep drifting on prefix for next event
                    for kk in ks.client_float_keys:
                        base_state[kk].copy_(self.bank.tensor_ref[kk])

                # NOTE: do not overwrite suffix in base_state here;
                # suffix will be overwritten by broadcast-to-all below.

        with torch.no_grad():
            if num_advanced > 0:
                inv_suf = 1.0 / float(k)
                for kk in ks.server_float_keys:
                    self.bank.server_state[kk].add_(inv_suf * acc_suf[kk])

            if num_ready > 0:
                inv_pre = 1.0 / float(num_ready)
                for kk in ks.client_float_keys:
                    self.bank.server_state[kk].add_(inv_pre * acc_pre[kk])

        self.server_version += 1

        with torch.no_grad():
            # suffix -> ALL every event
            for cid in range(self.num_clients):
                st = self.bank.client_base_state[cid]
                for kk in ks.server_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])
                self.client_state[cid].base_version = self.server_version

            # prefix -> READY only; reset progress; update anchors
            for cid in ready:
                self.client_state[cid].progress = 0.0
                st = self.bank.client_base_state[cid]
                for kk in ks.client_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])
                self.bank.copy_server_prefix_into_anchor(cid, ks)
                self.client_state[cid].prefix_version = self.server_version

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)

    def _run_async_event_dual_weighted(self) -> int:
        Hf = float(self.H)
        k = max(1, min(int(self.k), self.num_clients))

        ks = self.keysets
        if ks is None:
            self.keysets = self.bank.get_keysets(self.cut_idx)
            self.bank.ensure_prefix_anchor(self.keysets)
            ks = self.keysets
        assert ks is not None

        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, k)
        ready_set = set(ready)

        acc_pre = self.bank.tmp_base
        acc_suf = self.bank.tmp_delta
        assert acc_pre is not None and acc_suf is not None

        with torch.no_grad():
            for kk in ks.client_float_keys:
                acc_pre[kk].zero_()
            for kk in ks.server_float_keys:
                acc_suf[kk].zero_()

        num_ready = 0
        num_advanced = 0
        sum_steps = 0
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            sum_steps += nsteps
            if nsteps <= 0:
                continue
            num_advanced += 1

            base_state = self.bank.client_base_state[cid]
            anchor_state = self.bank.client_prefix_anchor_state[cid]
            self._local_train_from_base(cid, base_state, nsteps)

            with torch.no_grad():
                # keep local non-float / buffers
                for kk in self.bank.nonfloat_keys:
                    base_state[kk].copy_(self.bank.tensor_ref[kk])

                # suffix: sum ALL advanced contributions, divide by k later
                for kk in ks.server_float_keys:
                    acc_suf[kk].add_(self.bank.tensor_ref[kk] - base_state[kk])

                # prefix: READY only, delta contribution to fed server
                if cid in ready_set:
                    for kk in ks.client_float_keys:
                        acc_pre[kk].add_(self.bank.tensor_ref[kk] - anchor_state[kk])
                    num_ready += 1
                else:
                    # unfinished clients keep drifting on prefix for next event
                    for kk in ks.client_float_keys:
                        base_state[kk].copy_(self.bank.tensor_ref[kk])

                # NOTE: do not overwrite suffix in base_state here;
                # suffix will be overwritten by broadcast-to-all below.

        with torch.no_grad():
            if num_advanced > 0:
                #inv_suf = 1.0 / float(k)
                inv_suf = Hf / float(sum_steps)
                for kk in ks.server_float_keys:
                    self.bank.server_state[kk].add_(inv_suf * acc_suf[kk])

            if num_ready > 0:
                inv_pre = 1.0 / float(num_ready)
                for kk in ks.client_float_keys:
                    self.bank.server_state[kk].add_(inv_pre * acc_pre[kk])

        self.server_version += 1

        with torch.no_grad():
            # suffix -> ALL every event
            for cid in range(self.num_clients):
                st = self.bank.client_base_state[cid]
                for kk in ks.server_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])
                self.client_state[cid].base_version = self.server_version

            # prefix -> READY only; reset progress; update anchors
            for cid in ready:
                self.client_state[cid].progress = 0.0
                st = self.bank.client_base_state[cid]
                for kk in ks.client_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])
                self.bank.copy_server_prefix_into_anchor(cid, ks)
                self.client_state[cid].prefix_version = self.server_version

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)
    

    def _run_async_event_delta_modified(self) -> int:
        """
        Modified split baseline:
        - client-side (prefix): delta-buffer over READY clients only
        - server-side (suffix): direct parameter average over READY clients only
        - BOTH prefix and suffix are broadcast ONLY to READY clients
        - unfinished clients keep their local full model and continue next event
        """
        Hf = float(self.H)
        k = max(1, min(int(self.k), self.num_clients))

        ks = self.keysets
        if ks is None:
            self.keysets = self.bank.get_keysets(self.cut_idx)
            self.bank.ensure_prefix_anchor(self.keysets)
            ks = self.keysets
        assert ks is not None

        _, ready, steps_evt, _ = self._advance_to_k_complete_and_get_steps(Hf, k)
        ready_set = set(ready)

        acc_pre = self.bank.tmp_base   # prefix delta sum
        acc_suf = self.bank.tmp_delta  # suffix parameter sum
        assert acc_pre is not None and acc_suf is not None

        with torch.no_grad():
            for kk in ks.client_float_keys:
                acc_pre[kk].zero_()
            for kk in ks.server_float_keys:
                acc_suf[kk].zero_()

        num_ready = 0

        # 1) local training for all advanced clients
        for cid in range(self.num_clients):
            nsteps = int(steps_evt[cid])
            if nsteps <= 0:
                continue

            base_state = self.bank.client_base_state[cid]
            anchor_state = self.bank.client_prefix_anchor_state[cid]

            self._local_train_from_base(cid, base_state, nsteps)

            with torch.no_grad():
                # always keep local non-float / buffers
                for kk in self.bank.nonfloat_keys:
                    base_state[kk].copy_(self.bank.tensor_ref[kk])

                if cid in ready_set:
                    # prefix: delta-buffer against cycle anchor
                    for kk in ks.client_float_keys:
                        acc_pre[kk].add_(self.bank.tensor_ref[kk] - anchor_state[kk])

                    # suffix: direct parameter average over READY clients
                    for kk in ks.server_float_keys:
                        acc_suf[kk].add_(self.bank.tensor_ref[kk])

                    num_ready += 1
                else:
                    # unfinished clients keep drifting on BOTH prefix and suffix
                    for kk in ks.client_float_keys:
                        base_state[kk].copy_(self.bank.tensor_ref[kk])
                    for kk in ks.server_float_keys:
                        base_state[kk].copy_(self.bank.tensor_ref[kk])

        # 2) central update from READY clients only
        with torch.no_grad():
            if num_ready > 0:
                inv = 1.0 / float(num_ready)

                # client-side prefix: delta update
                for kk in ks.client_float_keys:
                    self.bank.server_state[kk].add_(inv * acc_pre[kk])

                # server-side suffix: direct average
                for kk in ks.server_float_keys:
                    self.bank.server_state[kk].copy_(inv * acc_suf[kk])

                self.server_version += 1

        # 3) sync ONLY READY clients to new central state
        with torch.no_grad():
            for cid in ready:
                self.client_state[cid].progress = 0.0

                st = self.bank.client_base_state[cid]

                # sync suffix
                for kk in ks.server_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])

                # sync prefix
                for kk in ks.client_float_keys:
                    st[kk].copy_(self.bank.server_state[kk])

                # ready clients start a new cycle from fresh central state
                self.bank.copy_server_prefix_into_anchor(cid, ks)
                self.client_state[cid].base_version = self.server_version
                self.client_state[cid].prefix_version = self.server_version

        self.global_phy_time += self._event_delay(ready, steps_evt)
        return len(ready)

    # ------------------------------------------------------------------
    # public loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, List[float]]:
        max_events = int(self.cfg.train.rounds)
        test_every = int(self.cfg.train.test_every)

        metrics = {
            "event_idx": [],
            "n_ready": [],
            "server_version": [],
            "phy_time": [],
            "test_acc": [],
            "test_loss": [],
            "staleness_event_idx": [],
            "staleness_avg": [],
            "staleness_max_so_far": [],
        }

        last_acc = float("nan")
        last_loss = float("nan")
        max_staleness_so_far = 0.0

        if bool(getattr(self.cfg.train, "eval_at_start", True)):
            last_acc, last_loss = self._evaluate()
            metrics["event_idx"].append(0)
            metrics["n_ready"].append(0)
            metrics["server_version"].append(self.server_version)
            metrics["phy_time"].append(self.global_phy_time)
            metrics["test_acc"].append(last_acc)
            metrics["test_loss"].append(last_loss)

        # record event 0 staleness
        st0 = self._current_staleness_vector()
        max_staleness_so_far = max(max_staleness_so_far, float(np.max(st0)))
        metrics["staleness_event_idx"].append(0)
        metrics["staleness_avg"].append(float(np.mean(st0)))
        metrics["staleness_max_so_far"].append(float(max_staleness_so_far))

        pbar = tqdm(range(1, max_events + 1), desc="Events")
        for event_idx in pbar:
            if self.scheme == 0:
                n_ready = self._run_sync_event()
            elif self.scheme == 1:
                n_ready = self._run_async_event_fedasync_hinge()
            elif self.scheme == 2:
                n_ready = self._run_async_event_fllike()
            elif self.scheme == 3:
                n_ready = self._run_async_event_dual()
            elif self.scheme == 4:
                n_ready = self._run_async_event_delta_fedbuff()
            elif self.scheme ==5:
                n_ready = self._run_async_event_dual_weighted()
            elif self.scheme == 6:
                n_ready = self._run_async_event_delta_modified()
            else:
                raise ValueError(f"Unsupported cfg.asyncenv.scheme: {self.scheme}")

            # record staleness every event
            st = self._current_staleness_vector()
            max_staleness_so_far = max(max_staleness_so_far, float(np.max(st)))
            metrics["staleness_event_idx"].append(event_idx)
            metrics["staleness_avg"].append(float(np.mean(st)))
            metrics["staleness_max_so_far"].append(float(max_staleness_so_far))

            if event_idx % test_every == 0:
                last_acc, last_loss = self._evaluate()
                metrics["event_idx"].append(event_idx)
                metrics["n_ready"].append(n_ready)
                metrics["server_version"].append(self.server_version)
                metrics["phy_time"].append(self.global_phy_time)
                metrics["test_acc"].append(last_acc)
                metrics["test_loss"].append(last_loss)

            pbar.set_postfix({
                "phy_time": f"{self.global_phy_time:.1f}s",
                "test_acc": f"{last_acc:.2f}%",
                "test_loss": f"{last_loss:.4f}",
                "stale_avg": f"{metrics['staleness_avg'][-1]:.2f}",
                "stale_max": f"{metrics['staleness_max_so_far'][-1]:.2f}",
            })

        return metrics