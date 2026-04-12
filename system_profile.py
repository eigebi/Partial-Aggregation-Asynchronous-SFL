# system_profile.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from models import build_model


# ============================================================
# 0. 基础数据结构
# ============================================================

@dataclass
class CutProfile:
    cut_layer: int
    flops_prefix: float
    flops_suffix: float
    bytes_smash_ul: float
    bytes_smashgrad_dl: float
    bytes_prefix_model: float
    bytes_suffix_model: float


@dataclass(frozen=True)
class ProfileSpec:
    model_name: str
    batch_size: int
    input_h: int
    input_w: int
    num_classes: int
    dtype_bytes: int
    profile_dir: str


# ============================================================
# 1. cfg -> spec / path
# ============================================================

def profile_spec_from_cfg(cfg) -> ProfileSpec:
    return ProfileSpec(
        model_name=str(cfg.model.name),
        batch_size=int(cfg.train.batch_size),
        input_h=int(cfg.model.profile_input_h),
        input_w=int(cfg.model.profile_input_w),
        num_classes=int(cfg.dataset.num_classes),
        dtype_bytes=int(cfg.model.profile_dtype_bytes),
        profile_dir=str(cfg.model.profile_dir),
    )


def build_profile_filename(
    model_name: str,
    batch_size: int,
    input_h: int,
    input_w: int,
) -> str:
    return f"{model_name}_bs{batch_size}_{input_h}x{input_w}.npy"


def build_profile_path(
    profile_dir: str,
    model_name: str,
    batch_size: int,
    input_h: int,
    input_w: int,
) -> str:
    os.makedirs(profile_dir, exist_ok=True)
    return os.path.join(
        profile_dir,
        build_profile_filename(model_name, batch_size, input_h, input_w),
    )


def resolve_profile_path_from_cfg(cfg) -> str:
    spec = profile_spec_from_cfg(cfg)
    return build_profile_path(
        profile_dir=spec.profile_dir,
        model_name=spec.model_name,
        batch_size=spec.batch_size,
        input_h=spec.input_h,
        input_w=spec.input_w,
    )


# ============================================================
# 2. Profile Bank
# ============================================================

class ProfileBank:
    """
    Generic profile bank.
    Kept backward-compatible with the old ResNetProfileBank usage style.
    """

    def __init__(
        self,
        npy_path: str,
        expected_model_name: Optional[str] = None,
        expected_batch_size: Optional[int] = None,
        expected_input_h: Optional[int] = None,
        expected_input_w: Optional[int] = None,
        expected_num_classes: Optional[int] = None,
    ):
        self.meta: Dict[str, Any] = {}
        self.profiles: Dict[int, CutProfile] = {}
        self._load_from_npy(
            path=npy_path,
            expected_model_name=expected_model_name,
            expected_batch_size=expected_batch_size,
            expected_input_h=expected_input_h,
            expected_input_w=expected_input_w,
            expected_num_classes=expected_num_classes,
        )

    def _load_from_npy(
        self,
        path: str,
        expected_model_name: Optional[str] = None,
        expected_batch_size: Optional[int] = None,
        expected_input_h: Optional[int] = None,
        expected_input_w: Optional[int] = None,
        expected_num_classes: Optional[int] = None,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile file not found: {path}")

        payload = np.load(path, allow_pickle=True).item()
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid profile file format in {path}")
        if "meta" not in payload or "profiles" not in payload:
            raise ValueError(f"Invalid profile payload in {path}: missing 'meta' or 'profiles'")

        self.meta = dict(payload["meta"])
        data_list = payload["profiles"]

        def _check(name: str, expected, cast_fn=int):
            if expected is None:
                return
            actual = cast_fn(self.meta.get(name, -1 if cast_fn is int else ""))
            if actual != cast_fn(expected):
                raise ValueError(
                    f"Profile {name} mismatch: file has {actual}, expected {expected}"
                )

        _check("model_name", expected_model_name, str)
        _check("batch_size", expected_batch_size, int)
        _check("input_h", expected_input_h, int)
        _check("input_w", expected_input_w, int)
        _check("num_classes", expected_num_classes, int)

        self.profiles.clear()
        for p in data_list:
            cut_idx = int(p["cut_layer"])
            self.profiles[cut_idx] = CutProfile(
                cut_layer=cut_idx,
                flops_prefix=float(p["flops_prefix"]),
                flops_suffix=float(p["flops_suffix"]),
                bytes_smash_ul=float(p["bytes_smash_ul"]),
                bytes_smashgrad_dl=float(p["bytes_smashgrad_dl"]),
                bytes_prefix_model=float(p["bytes_prefix_model"]),
                bytes_suffix_model=float(p["bytes_suffix_model"]),
            )

        print(
            f"[ProfileBank] Loaded {len(self.profiles)} cut profiles from {path}\n"
            f"[ProfileBank] Meta: {self.meta}"
        )

    def get(self, cut_layer: int) -> CutProfile:
        if cut_layer not in self.profiles:
            raise ValueError(
                f"Cut layer {cut_layer} not found. Available cuts: {list(self.profiles.keys())}"
            )
        return self.profiles[cut_layer]

    def get_all_cuts(self) -> List[int]:
        return sorted(self.profiles.keys())

    def get_meta(self) -> Dict[str, Any]:
        return dict(self.meta)


# backward compatibility
ResNetProfileBank = ProfileBank


# ============================================================
# 3. Profiler
# ============================================================

class CustomSFLProfiler:
    """
    Profile all cut points of the model.
    The model itself is responsible for how a cut is split into client/server modules.
    """

    def __init__(
        self,
        model_name: str = "resnet34",
        num_classes: int = 10,
        batch_size: int = 16,
        input_h: int = 32,
        input_w: int = 32,
        dtype_bytes: int = 4,
        device: str = "cpu",
    ):
        self.model_name = str(model_name)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.dtype_bytes = int(dtype_bytes)
        self.device = torch.device(device)

        self.input_shape = (self.batch_size, 3, self.input_h, self.input_w)

        # build_model currently decides stem type based on input_size
        # use max(h, w) so 224x224-like settings select the correct stem
        self.reference_model = build_model(
            arch=self.model_name,
            num_classes=self.num_classes,
            input_size=max(self.input_h, self.input_w),
        ).to(self.device).eval()

        if not hasattr(self.reference_model, "num_blocks"):
            raise AttributeError("Model must implement num_blocks()")
        if not hasattr(self.reference_model, "build_profile_modules"):
            raise AttributeError("Model must implement build_profile_modules(cut_idx)")

        self.num_blocks = int(self.reference_model.num_blocks())

        print(
            f"[Profiler] Model={self.model_name}, num_classes={self.num_classes}, "
            f"input_shape={self.input_shape}, total_blocks={self.num_blocks}"
        )

    def _count_params_bytes(self, module: nn.Module) -> float:
        num_params = sum(p.numel() for p in module.parameters())
        return float(num_params * self.dtype_bytes)

    def profile_cut(self, cut_idx: int) -> Optional[Dict[str, float]]:
        if not (0 <= cut_idx <= self.num_blocks):
            return None

        client_model, server_model = self.reference_model.build_profile_modules(cut_idx)
        client_model = client_model.to(self.device).eval()
        server_model = server_model.to(self.device).eval()

        dummy_input = torch.randn(self.input_shape, device=self.device)

        try:
            client_macs, _ = profile(client_model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            print(f"[Profiler] Error profiling client cut {cut_idx}: {e}")
            return None

        client_flops = float(2.0 * client_macs)

        with torch.no_grad():
            smashed_data = client_model(dummy_input)

        bytes_smash = float(smashed_data.numel() * self.dtype_bytes)

        try:
            server_macs, _ = profile(server_model, inputs=(smashed_data,), verbose=False)
        except Exception as e:
            print(f"[Profiler] Error profiling server cut {cut_idx}: {e}")
            return None

        server_flops = float(2.0 * server_macs)

        bytes_prefix = self._count_params_bytes(client_model)
        bytes_suffix = self._count_params_bytes(server_model)

        return {
            "cut_layer": int(cut_idx),
            "flops_prefix": client_flops,
            "flops_suffix": server_flops,
            "bytes_smash_ul": bytes_smash,
            "bytes_smashgrad_dl": bytes_smash,
            "bytes_prefix_model": bytes_prefix,
            "bytes_suffix_model": bytes_suffix,
        }

    def run_all(self) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []

        print(
            f"{'Cut':<5} | {'C-FLOPs(G)':<12} | {'S-FLOPs(G)':<12} | "
            f"{'Smash(MB)':<10} | {'C-Model(MB)':<12}"
        )
        print("-" * 72)

        for cut_idx in range(self.num_blocks + 1):
            res = self.profile_cut(cut_idx)
            if res is None:
                continue
            results.append(res)
            print(
                f"{cut_idx:<5} | "
                f"{res['flops_prefix']/1e9:<12.4f} | "
                f"{res['flops_suffix']/1e9:<12.4f} | "
                f"{res['bytes_smash_ul']/1e6:<10.4f} | "
                f"{res['bytes_prefix_model']/1e6:<12.4f}"
            )

        return results


# ============================================================
# 4. Save / generate / ensure
# ============================================================

def save_profiles(
    profile_path: str,
    profiles: List[Dict[str, float]],
    spec: ProfileSpec,
) -> None:
    payload = {
        "meta": {
            "model_name": spec.model_name,
            "batch_size": spec.batch_size,
            "input_h": spec.input_h,
            "input_w": spec.input_w,
            "num_classes": spec.num_classes,
            "dtype_bytes": spec.dtype_bytes,
        },
        "profiles": profiles,
    }
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    np.save(profile_path, payload, allow_pickle=True)
    print(f"[Profile] Saved all profiles to: {profile_path}")


def generate_and_save_profiles_from_cfg(cfg, device: str = "cpu") -> str:
    spec = profile_spec_from_cfg(cfg)
    profile_path = resolve_profile_path_from_cfg(cfg)

    profiler = CustomSFLProfiler(
        model_name=spec.model_name,
        num_classes=spec.num_classes,
        batch_size=spec.batch_size,
        input_h=spec.input_h,
        input_w=spec.input_w,
        dtype_bytes=spec.dtype_bytes,
        device=device,
    )
    profiles = profiler.run_all()
    save_profiles(profile_path, profiles, spec)
    return profile_path


def ensure_profile_file_from_cfg(
    cfg,
    device: str = "cpu",
    force_regen: bool = False,
) -> str:
    profile_path = resolve_profile_path_from_cfg(cfg)
    if force_regen or (not os.path.exists(profile_path)):
        print(f"[Profile] Missing profile file, generating: {profile_path}")
        generate_and_save_profiles_from_cfg(cfg, device=device)
    else:
        print(f"[Profile] Found existing profile file: {profile_path}")
    return profile_path


def ensure_profile_bank_from_cfg(
    cfg,
    device: str = "cpu",
    force_regen: bool = False,
) -> ProfileBank:
    profile_path = ensure_profile_file_from_cfg(
        cfg=cfg,
        device=device,
        force_regen=force_regen,
    )
    spec = profile_spec_from_cfg(cfg)
    return ProfileBank(
        npy_path=profile_path,
        expected_model_name=spec.model_name,
        expected_batch_size=spec.batch_size,
        expected_input_h=spec.input_h,
        expected_input_w=spec.input_w,
        expected_num_classes=spec.num_classes,
    )


# backward-compatible name
def load_profile_bank_from_cfg(cfg, device: str = "cpu", force_regen: bool = False) -> ProfileBank:
    return ensure_profile_bank_from_cfg(cfg=cfg, device=device, force_regen=force_regen)


# ============================================================
# 5. Standalone entry
# ============================================================

if __name__ == "__main__":
    try:
        from main import build_config
        cfg = build_config()
        print("[ProfileMain] Loaded cfg from main.build_config()")
    except Exception:
        from config import ExperimentCfg
        cfg = ExperimentCfg()
        print("[ProfileMain] Falling back to default ExperimentCfg()")

    bank = ensure_profile_bank_from_cfg(cfg, device="cpu", force_regen=False)
    print("[ProfileMain] Meta:", bank.get_meta())
    print("[ProfileMain] Available cuts:", bank.get_all_cuts())
    if bank.get_all_cuts():
        cut0 = bank.get_all_cuts()[min(1, len(bank.get_all_cuts()) - 1)]
        prof = bank.get(cut0)
        print(
            f"[ProfileMain] Example cut={cut0}: "
            f"C-FLOPs={prof.flops_prefix/1e9:.4f}G, "
            f"S-FLOPs={prof.flops_suffix/1e9:.4f}G, "
            f"Smash={prof.bytes_smash_ul/1e6:.4f}MB"
        )