# main.py
from __future__ import annotations

import os
import argparse
import random
import numpy as np
import torch
import yaml

from post_msg import send_wechat_msg
from config import ExperimentCfg

from datasets import get_datasets
from dirichlet_split import split_clients_dirichlet
from system_profile import resolve_profile_path_from_cfg
from models import build_model, ClientModelBank
from engine import SplitFedEngine
from async_env import sample_clients_correlated_lognormal, EdgeCaps


# ============================================================
# 1. build configuration
# ============================================================

def _split_cli_values(raw_items):
    vals = []
    if raw_items is None:
        return vals
    for item in raw_items:
        for part in str(item).split(","):
            part = part.strip()
            if part != "":
                vals.append(part)
    return vals


def _resolve_cfg_path(cfg, path: str):
    cur = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise AttributeError(f"Invalid config path: {path} (missing '{p}')")
        cur = getattr(cur, p)

    leaf = parts[-1]
    if not hasattr(cur, leaf):
        raise AttributeError(f"Invalid config path: {path} (missing '{leaf}')")
    return cur, leaf


def _cast_cli_value(raw: str, ref_value):
    if isinstance(ref_value, bool):
        s = raw.lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Cannot parse bool from '{raw}'")

    if isinstance(ref_value, int) and not isinstance(ref_value, bool):
        return int(raw)

    if isinstance(ref_value, float):
        return float(raw)

    return raw


def _apply_override(cfg, expr: str) -> None:
    if "=" not in expr:
        raise ValueError(f"Bad override '{expr}', expected key=value")
    key, raw = expr.split("=", 1)
    target_obj, target_attr = _resolve_cfg_path(cfg, key)
    ref_value = getattr(target_obj, target_attr)
    value = _cast_cli_value(raw, ref_value)
    setattr(target_obj, target_attr, value)


def build_config():
    parser = argparse.ArgumentParser(description="实验配置")

    # YAML
    parser.add_argument("--config", type=str, default="./default_config.yaml", help="YAML配置文件路径")

    # legacy sweep mode
    parser.add_argument("--name_para", type=str, default=None, help="待测试的参数名称，如 asyncenv.scheme")
    parser.add_argument("--value_para", type=str, nargs="+", default=None, help="待测试的参数值，可写成 1 2 3 或 1,2,3")
    parser.add_argument("--num_runs", type=int, default=None, help="运行次数")

    # job mode
    parser.add_argument("--run_id", type=int, default=None, help="只运行某一个 run_id")
    parser.add_argument("--tag", type=str, default=None, help="实验分组标签")
    parser.add_argument("--job_name", type=str, default=None, help="当前 job 名称")
    parser.add_argument("--out_root", type=str, default="./results/conference", help="结果根目录")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config by key=value, e.g. asyncenv.scheme=3",
    )

    args = parser.parse_args()
    cfg = ExperimentCfg()

    # 1) load YAML first
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            cfg = ExperimentCfg.load_from_dict(yaml_data)
            print(f"[Config] 已从 {args.config} 加载配置")

    # 2) legacy sweep CLI override
    if args.name_para is not None:
        cfg.name_para = args.name_para

    if args.value_para is not None:
        cfg.value_para = _split_cli_values(args.value_para)

    if args.num_runs is not None:
        cfg.num_runs = args.num_runs

    # 3) job-mode run selector
    if args.run_id is not None:
        cfg.start_run = int(args.run_id)
        cfg.num_runs = 1

    # 4) generic key=value overrides
    for expr in args.overrides:
        _apply_override(cfg, expr)

    # defaults only if still missing
    if getattr(cfg, "name_para", None) is None:
        cfg.name_para = "asyncenv.scheme"
    if getattr(cfg, "value_para", None) is None:
        cfg.value_para = ["2"]

    return cfg, args


# ============================================================
# 2. build random instances for a run
# ============================================================

def build_run_instances(cfg: ExperimentCfg, train_set, seed: int):
    """
    生成每个 run 内存在随机性的实体：
    1) client_splits
    2) client_caps
    3) edge_caps
    """
    edge_caps = EdgeCaps(
        total_flops=float(cfg.system.edge_flops * 1e9),
        ulf_mbps=float(cfg.system.edge_ulf),
        dlf_mbps=float(cfg.system.edge_dlf),
    )

    client_caps = sample_clients_correlated_lognormal(
        n=int(cfg.asyncenv.num_clients),
        ul_range_mbps=tuple(cfg.system.ul_range),
        dl_range_mbps=tuple(cfg.system.dl_range),
        flops_range=(
            float(cfg.system.flops_range[0] * 1e9),
            float(cfg.system.flops_range[1] * 1e9),
        ),
        seed=seed,
        corr_strength=float(cfg.system.corr_strength),
    )

    client_splits = split_clients_dirichlet(
        dataset=train_set,
        num_clients=int(cfg.asyncenv.num_clients),
        beta=float(cfg.dataset.beta),
        min_size=cfg.train.batch_size * 5,  # 保证每个 client 至少有5个 batch
        seed=seed,
    )

    return client_splits, client_caps, edge_caps


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def run_experiment(
    cfg: ExperimentCfg,
    train_set,
    test_set,
    client_splits,
    client_caps,
    edge_caps: EdgeCaps,
    seed: int,
):
    """
    跑单个配置。
    这里只用函数传进来的 train/test set，不依赖任何全局变量。
    """
    set_global_seed(seed, deterministic=cfg.train.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        arch=str(cfg.model.name),
        num_classes=int(cfg.dataset.num_classes),
        input_size=int(cfg.model.profile_input_w),
    ).to(device)

    bank = ClientModelBank(
        num_clients=int(cfg.asyncenv.num_clients),
        model=model,
        device=device,
    )

    engine = SplitFedEngine(
        cfg=cfg,
        bank=bank,
        train_set=train_set,
        test_set=test_set,
        client_splits=client_splits,
        client_caps=client_caps,
        edge_caps=edge_caps,
        initial_cut_idx=cfg.asyncenv.init_cut_idx,
        runtime_seed=seed,
    )

    metrics = engine.run()
    return metrics


# ============================================================
# 3. main
# ============================================================

def main():
    cfg, cli = build_config()

    # job mode
    if cli.job_name is not None:
        tag = cli.tag if cli.tag is not None else "adhoc"
        results_dir = os.path.join(cli.out_root, tag, cli.job_name)
        job_mode = True
        print(f"=== Running Job: tag={tag}, job={cli.job_name} ===")
    else:
        # legacy sweep mode
        results_dir = os.path.join(
            cli.out_root,
            f"para={cfg.name_para}_value={cfg.value_para}",
        )
        job_mode = False
        print(f"=== Running Sweep: {cfg.name_para} = {cfg.value_para} ===")

    os.makedirs(results_dir, exist_ok=True)
    cfg.save_to_yaml(os.path.join(results_dir, "running_config.yaml"))

    print(f"=== Loading Datasets: {cfg.dataset.name} ===")
    train_set, test_set = get_datasets(
        cfg.dataset.name,
        cfg.dataset.dir,
    )

    seed = cfg.seed
    set_global_seed(seed, deterministic=cfg.train.deterministic)

    for run_id in range(cfg.start_run, cfg.start_run + cfg.num_runs):
        run_seed = seed + run_id * 100
        client_seed = seed if cfg.system.fixed_clients_stats else run_seed
        client_splits, client_caps, edge_caps = build_run_instances(cfg, train_set, client_seed)

        if job_mode:
            print(f"\n>>> Running job @ Run {run_id}")
            print(
                f"    scheme={cfg.asyncenv.scheme}, "
                f"min_ready={cfg.asyncenv.min_ready_clients}, "
                f"beta={cfg.dataset.beta}, "
                f"cut={cfg.asyncenv.init_cut_idx}, "
                f"model={cfg.model.name}, "
                f"rounds={cfg.train.rounds}"
            )

            metrics = run_experiment(
                cfg=cfg,
                train_set=train_set,
                test_set=test_set,
                client_splits=client_splits,
                client_caps=client_caps,
                edge_caps=edge_caps,
                seed=run_seed,
            )

            metrics_path = os.path.join(results_dir, f"metrics_run{run_id}.npz")
            save_dict = {k: np.array(v, dtype=np.float32) for k, v in metrics.items()}
            np.savez(metrics_path, **save_dict)
            print(f">>> Saved metrics: {metrics_path}")

            if "test_acc" in metrics and len(metrics["test_acc"]) > 0:
                final_acc = float(metrics["test_acc"][-1])
                print(f"    Final Acc: {final_acc:.2f}%")
                send_wechat_msg(f"[{tag}/{cli.job_name}] run {run_id} finished. Final Acc: {final_acc:.2f}%")
            else:
                print("    Warning: metrics['test_acc'] missing or empty.")
                send_wechat_msg(f"[{tag}/{cli.job_name}] run {run_id} finished.")

        else:
            for para_id, para_value_raw in enumerate(cfg.value_para):
                target_obj, target_attr = _resolve_cfg_path(cfg, cfg.name_para)
                ref_value = getattr(target_obj, target_attr)
                para_value = _cast_cli_value(str(para_value_raw), ref_value)
                setattr(target_obj, target_attr, para_value)

                print(f"\n>>> Running Sweep Parameter: {cfg.name_para}={para_value} @ Run {run_id}")

                para_seed = run_seed + para_id * 1000
                _ = resolve_profile_path_from_cfg(cfg)

                metrics = run_experiment(
                    cfg=cfg,
                    train_set=train_set,
                    test_set=test_set,
                    client_splits=client_splits,
                    client_caps=client_caps,
                    edge_caps=edge_caps,
                    seed=para_seed,
                )

                metrics_path = os.path.join(results_dir, f"{cfg.name_para}={para_value}_run{run_id}.npz")
                save_dict = {k: np.array(v, dtype=np.float32) for k, v in metrics.items()}
                np.savez(metrics_path, **save_dict)
                print(f">>> Saved metrics: {metrics_path}")

                if "test_acc" in metrics and len(metrics["test_acc"]) > 0:
                    final_acc = float(metrics["test_acc"][-1])
                    print(f"    Final Acc: {final_acc:.2f}%")
                    send_wechat_msg(f"Run {run_id} / {cfg.name_para}={para_value} finished. Final Acc: {final_acc:.2f}%")
                else:
                    print("    Warning: metrics['test_acc'] missing or empty.")
                    send_wechat_msg(f"Run {run_id} / {cfg.name_para}={para_value} finished.")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()