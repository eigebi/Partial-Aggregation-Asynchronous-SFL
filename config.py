# config.py
# 这个文件定义了实验配置的结构和加载逻辑
# 优先级：默认值 < YAML文件 < 命令行参数
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional
import yaml, argparse, os

@dataclass
class DatasetCfg:
    name: str = "cifar10"
    dir: str = "./data"
    num_classes: int = 10
    #valid_ratio: float = 0.0
    beta: Optional[float] = 0.1 # Dirichlet分布的beta参数，越小越不均匀
    min_client_samples: int = 10 # 可以设置成n倍batch_size，保证每个client至少有若干个batch

@dataclass
class ModelCfg:
    name: str = "resnet18"
    num_layers: int = 17 # 待定，有效信息太少了，可能要跟别的合并
    #cut_idx: int = 0 # 这个可能要删掉，最多有一个initil放在系统配置里
    #norm: str = "bn"
    #gn_groups: int = 32
    # Profile
    profile_dir: str = "./profiler"
    #profile_batch_size: int = 16 # 应该放在train里
    profile_input_h: int = 32
    profile_input_w: int = 32
    #profile_num_classes: int = 10
    profile_dtype_bytes: int = 4
    #profile_path: str = ""

@dataclass
class SystemCfg:
    # Client Caps Params (Range: [min, max])
    ul_range: list[float] = field(default_factory=lambda: [2.0, 20.0])   # Mbps (Client->Edge)
    dl_range: list[float] = field(default_factory=lambda: [10.0, 50.0])  # Mbps (Edge->Client)
    flops_range: list[float] = field(default_factory=lambda: [20.0, 500.0]) # GFLOPs (Client Compute)
    corr_strength: float = 0.8  # 算力与带宽的相关性 (0~1)
    fixed_clients_stats: bool = False # 是否生成一样的用户组和数据集（二者绑定）
    
    # Edge Caps Params
    edge_flops: float = 20000.0 # GFLOPs (20 TFLOPs)
    edge_ulf: float = 400.0    # Mbps (Edge->Fed)
    edge_dlf: float = 400.0    # Mbps (Fed->Edge)


@dataclass
class AsyncEnvCfg: # 改了名字，原来是AsyncCfg
    #scheme: Literal["sync", "async_fllike", "async_dual"] = "sync"
    scheme: int = 2 # 0: sync, 1: FedAsync, 2: FedBuffer (min_ready>1), 3: Proposed 
    #agg_scheme: int = 0 # 0: fl_like, 1: dual_pace, 2: possibly sfl, 
    delay_opt: int = 0 # 0: min_k_ready, fixed_cut, 1: per_round_opt, 2:staleness_control
    client_cotrain_steps: int = 20
    max_staleness: int = 50 # 可能要删掉，或者接入staleness control的expected staleness constraint
    #acceptance: Literal["accept", "drop", "weight"] = "weight"
    #staleness_weight: Literal["inv", "exp"] = "inv" # 留着后续对比接口，因为semi那篇用了衰减函数weight stale gradient
    exp_gamma: float = 0.95
    #lr_client_scale: float = 1.0 # 没用
    #lr_server_scale: float = 1.0
    num_clients: int = 10
    min_ready_clients: int = 5
    init_cut_idx: int = 4 
    #V: float = 1.0 # Lyapunov control的V参数

@dataclass
class TrainCfg:
    #num_clients: int = 5 # 这个放在AsyncCfg里更合适
    rounds: int = 500
    #frac: float = 1.0 # 
    batch_size: int = 32
    num_workers: int = 2 # DataLoader的num_workers
    lr: float = 0.005
    momentum: float = 0 # optimizer的momentum，永远是0
    weight_decay: float = 5e-4 # optimizer相关
    test_every: int = 5
    test_batch_size: int = 512
    #seed: int = 1234 # 放最外面
    deterministic: bool = True
    debug_repro_mode: bool = False
    #loader_seed_offset: int = 1000
    #sync_choice_seed_offset: int = 2000
    #eval_seed_offset: int = 3000

@dataclass
class ExperimentCfg:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    system: SystemCfg = field(default_factory=SystemCfg) 
    asyncenv: AsyncEnvCfg = field(default_factory=AsyncEnvCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    #extra: Dict[str, Any] = field(default_factory=dict)
    seed: int = 2026
    start_run: int = 0
    num_runs: int = 5
    
    save_fingerprint: bool = True
    save_debug_trace: bool = True
    
    name_para: Optional[str] = None
    value_para: Optional[Any] = None

    def save_to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, sort_keys=False)

    @staticmethod
    def load_from_dict(d: dict) -> ExperimentCfg:
        cfg = ExperimentCfg()
        for key, value in d.items():
            if hasattr(cfg, key):
                target = getattr(cfg, key)
                if hasattr(target, "__dataclass_fields__") and isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        setattr(target, sub_k, sub_v)
                else:
                    setattr(cfg, key, value)
        return cfg



if __name__ == "__main__":
    from main import build_config
    cfg = build_config()