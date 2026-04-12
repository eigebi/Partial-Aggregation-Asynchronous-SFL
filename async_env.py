# system_async_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# 必须有这个 ProfileBank 和 CutProfile
from system_profile import CutProfile, ProfileBank

# ==========================================
# 1. 基础能力定义 (Caps)
# ==========================================
@dataclass(frozen=True)
class ClientCaps:
    ul_mbps: float       # Client -> Edge 上行带宽 (Mbps)
    dl_mbps: float       # Edge -> Client 下行带宽 (Mbps)
    ulf_mbps: float      # Client -> Fed  上行带宽 (Mbps)
    dlf_mbps: float      # Fed -> Client  下行带宽 (Mbps)
    flops: float         # 客户端算力 (FLOPs/s)

@dataclass(frozen=True)
class EdgeCaps:
    total_flops: float   # 边缘服务器总算力 (FLOPs/s)
    ulf_mbps: float      # Edge -> Fed 上行带宽 (Mbps)
    dlf_mbps: float      # Fed -> Edge 下行带宽 (Mbps)


# ==========================================
# 2. 辅助工具函数
# ==========================================
def _clip_pos(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """防止除以0"""
    return np.maximum(x, eps)

def _mbps_to_bps(mbps: float) -> float:
    return float(mbps) * 1e6

def _bytes_to_bits(num_bytes: float) -> float:
    return float(num_bytes) * 8.0


# ==========================================
# 3. 客户端生成器
# ==========================================
def sample_clients_correlated_lognormal(
    n: int,
    ul_range_mbps: Tuple[float, float],
    dl_range_mbps: Tuple[float, float],
    flops_range: Tuple[float, float],
    seed: int = 0,
    corr_strength: float = 1.0,
) -> List[ClientCaps]:
    """
    生成 n 个异构客户端。
    带宽和算力具有相关性 (corr_strength)。
    """
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=n)
    noise = rng.normal(size=n)
    
    # 混合噪声生成相关性
    z2 = corr_strength * z + (1.0 - corr_strength) * noise

    # 归一化 rank 到 [0,1]
    r_flops = (z - z.min()) / max(z.max() - z.min(), 1e-12)
    r_bw = (z2 - z2.min()) / max(z2.max() - z2.min(), 1e-12)

    def map_log_range(r, lo, hi):
        lo = max(float(lo), 1e-12)
        hi = max(float(hi), lo * 1.0001)
        return np.exp(np.log(lo) + r * (np.log(hi) - np.log(lo)))

    flops = map_log_range(r_flops, flops_range[0], flops_range[1])
    ul = map_log_range(r_bw, ul_range_mbps[0], ul_range_mbps[1])
    dl = map_log_range(r_bw, dl_range_mbps[0], dl_range_mbps[1])

    # 这里假设连接 Fed 的带宽与连接 Edge 的带宽同分布初始化
    # 如果你想让它们不同，可以在这里修改逻辑
    return [
        ClientCaps(
            ul_mbps=float(ul[i]), 
            dl_mbps=float(dl[i]), 
            ulf_mbps=float(ul[i]),  # 假设连Fed和连Edge带宽一致
            dlf_mbps=float(dl[i]), 
            flops=float(flops[i])
        ) 
        for i in range(n)
    ]


# ==========================================
# 4. 核心环境 (AsyncSystemEnv)
# ==========================================
class AsyncSystemEnv:
    """
    物理模拟器：将 (策略, 资源, 模型物理量) -> 转化为 -> (时延)
    """

    def __init__(self, clients: List[ClientCaps], edge: EdgeCaps, profile_bank: ProfileBank):
        self.clients = clients
        self.edge = edge
        self.profile_bank: ProfileBank = profile_bank  # 需要外部加载并传入
        self.alloc_edge_flops: bool = False  # 是否启用 Edge FLOPs 分配机制; 会议版暂时平均分配

    def event_latency(
        self,
        client_cut_layers: np.ndarray,    # [输入1] 策略：每个 Client 的 cut layer index (shape [N])
        H_vec: np.ndarray,                # [输入2] 策略：训练步数 (shape [N])
        ready_vec: List[int],           # [输入3] 策略：指定聚合名单 (必须提供)
        edge_alloc_norm: np.ndarray = None,      # [输入4] 策略：Edge 资源分配 (shape [N])
        
        # 物理常数 (Backward = 2 * Forward) n+1
        mul_client_fb: float = 3.0,       
        mul_edge_fb: float = 3.0,      
    ) -> Dict[str, float]:
        
        N = len(self.clients)
        
        # 参数校验
        if ready_vec is None:
            raise ValueError("ready set must be provided by strategy.")
        if self.alloc_edge_flops:
            edge_alloc_norm = np.asarray(edge_alloc_norm, dtype=np.float64)
        else:
            edge_alloc_norm = np.ones(N, dtype=np.float64)  # 平均分配
        H_vec = np.asarray(H_vec, dtype=np.float64)
        client_cut_layers = np.asarray(client_cut_layers, dtype=np.int32)

        # -------------------------------------------------
        # Step 1: 物理速率计算 (Edge 资源分配)
        # -------------------------------------------------
        # 归一化分配比例
        a = np.maximum(edge_alloc_norm, 0.0)
        s = float(a.sum())
        if s <= 0.0:
            a = np.ones(N, dtype=np.float64) / float(N)
        else:
            a = a / s
        
        # 每个人分到的实际 Edge 算力 (FLOPs/s)
        edge_flops_i = _clip_pos(a * float(self.edge.total_flops))

        # -------------------------------------------------
        # Step 2: 单步时延计算 (Latency per Step)
        # -------------------------------------------------
        t_step = np.zeros(N, dtype=np.float64)
        
        # 预先提取列表以加速循环
        c_flops_list = [c.flops for c in self.clients]
        c_ul_bps_list = [_mbps_to_bps(c.ul_mbps) for c in self.clients]
        c_dl_bps_list = [_mbps_to_bps(c.dl_mbps) for c in self.clients]

        for i in range(N):
            # 1. 查表：获取该 Client 在当前 Cut 下的物理量
            cut_idx = int(client_cut_layers[i])
            prof = self.profile_bank.get(cut_idx) 

            # 2. 提取物理属性
            flops_pre = float(prof.flops_prefix)
            flops_suf = float(prof.flops_suffix)
            bits_smash = _bytes_to_bits(float(prof.bytes_smash_ul))
            bits_grad  = _bytes_to_bits(float(prof.bytes_smashgrad_dl))

            # 3. 计算四段流水线时间
            # Client 计算
            t_client_comp = float(mul_client_fb) * flops_pre / _clip_pos(np.array([c_flops_list[i]]))[0]
            # Edge 计算
            t_edge_comp   = float(mul_edge_fb) * flops_suf / edge_flops_i[i]
            # Smash Data 上传
            t_ul          = bits_smash / _clip_pos(np.array([c_ul_bps_list[i]]))[0]
            # Gradient 下载
            t_dl          = bits_grad / _clip_pos(np.array([c_dl_bps_list[i]]))[0]

            t_step[i] = t_client_comp + t_edge_comp + t_ul + t_dl

        # -------------------------------------------------
        # Step 3: 确定训练完成时间 (Trigger Time)
        # -------------------------------------------------
        # 每个client都要跑 Hn 步
        t_finish = H_vec * t_step 

        # 过滤非法索引
        ready_indices = [int(i) for i in ready_vec if 0 <= i < N]

        if len(ready_indices) == 0:
             return {"T_event": 0.0}

        # 训练触发时间 = Ready 集合里最慢的那个 (Barrier)
        t_trigger = float(np.max(t_finish[ready_indices]))

        # -------------------------------------------------
        # Step 4: 模型聚合 (Edge-Fed & Client-Fed)
        # -------------------------------------------------
        load_clients_list = []
        ready_ulf_bps = []
        ready_dlf_bps = []

        for idx in ready_indices:
            # 获取该 Client 的模型大小 (Prefix)
            prof = self.profile_bank.get(int(client_cut_layers[idx]))
            load_clients_list.append(_bytes_to_bits(float(prof.bytes_prefix_model)))
            
            # 获取连接 Fed 的带宽
            ready_ulf_bps.append(_mbps_to_bps(self.clients[idx].ulf_mbps))
            ready_dlf_bps.append(_mbps_to_bps(self.clients[idx].dlf_mbps))

        load_clients = np.array(load_clients_list, dtype=np.float64)
        ready_ulf_bps = np.array(ready_ulf_bps, dtype=np.float64)
        ready_dlf_bps = np.array(ready_dlf_bps, dtype=np.float64)

        # [核心逻辑] Edge 负载 = (人数 * 最大模型) - 客户端模型总和
        # Edge 负责补齐 Common 部分的差异
        load_edge = float(len(ready_indices) * np.max(load_clients) - np.sum(load_clients))

        # Edge 连接 Fed 的带宽
        edge_ulf_bps = _clip_pos(np.array([_mbps_to_bps(self.edge.ulf_mbps)]))[0]
        edge_dlf_bps = _clip_pos(np.array([_mbps_to_bps(self.edge.dlf_mbps)]))[0]

        t_edge2fed = load_edge / edge_ulf_bps
        t_fed2edge = load_edge / edge_dlf_bps

        # 1. 提取每个被选中客户端的独立时间
        t_comp_ready = t_finish[ready_indices] # 也就是 H_n * t_step_n
        t_ul_ready = load_clients / _clip_pos(ready_ulf_bps)
        t_dl_ready = load_clients / _clip_pos(ready_dlf_bps)
        
        # 2. 计算最慢的计算时间 (Edge必须等这个时间点才能开始汇总并上传)
        T_comp_max = float(np.max(t_comp_ready)) if len(ready_indices) > 0 else 0.0
        
        # 3. 客户端单兵流水线的最晚结束时间
        T_client_pipeline_max = float(np.max(t_comp_ready + t_ul_ready)) if len(ready_indices) > 0 else 0.0
        
        # 4. Edge 补齐与上传时间
        t_edge2fed = float(load_edge / edge_ulf_bps)
        t_fed2edge = float(load_edge / edge_dlf_bps)
        
        # 5. 【核心】整个并发流水线阶段的真实耗时
        T_pipeline = max(T_client_pipeline_max, T_comp_max + t_edge2fed)
        
        # 6. 全局下载阶段屏障
        T_dl_max_clients = float(np.max(t_dl_ready)) if len(ready_indices) > 0 else 0.0
        T_dl = max(T_dl_max_clients, t_fed2edge)

        # 真实物理流逝总时间
        T_event = T_pipeline + T_dl

        return {
            "T_event": float(T_event),
            "T_pipeline": T_pipeline, # <--- 传出这个核心变量
            "T_comp_max": T_comp_max,
            "T_dl": T_dl
        }
    
def summarize_client_caps(client_caps):
    return {
        "num_clients": len(client_caps),
        "ul_mbps": [float(c.ul_mbps) for c in client_caps],
        "dl_mbps": [float(c.dl_mbps) for c in client_caps],
        "client_tflops": [float(c.client_tflops) for c in client_caps],
        "ulf": [float(c.ulf) for c in client_caps],
        "dlf": [float(c.dlf) for c in client_caps],
    }

# ==========================================
# 5. 测试入口 (独立运行测试)
# ==========================================
if __name__ == "__main__":
    import os
    
    # 路径检查
    PROFILE_PATH = "./profiler/resnet34_bs32_32x32.npy"
    if not os.path.exists(PROFILE_PATH):
        print(f"[Warn] {PROFILE_PATH} 不存在，无法运行测试。")
        exit()

    # 1. 加载库
    bank = ProfileBank(PROFILE_PATH)
    
    # 2. 初始化 Clients (10个)
    NUM_CLIENTS = 10
    clients = sample_clients_correlated_lognormal(
        n=NUM_CLIENTS, 
        ul_range_mbps=(2.0, 20.0), 
        dl_range_mbps=(10.0, 50.0), 
        flops_range=(10e9, 100e9),
        seed=42
    )
    
    # 3. 初始化 Edge
    edge = EdgeCaps(
        total_flops=50e12, 
        ulf_mbps=2000.0, 
        dlf_mbps=2000.0  
    )
    
    env = AsyncSystemEnv(clients, edge, bank)

    print(f"\n>>> 环境测试启动")

    # 4. 构造输入策略 (Mock Strategy)
    # (A) Cut: 前5个在 Layer 3, 后5个在 Layer 8
    cuts = np.array([3]*5 + [8]*5) 
    # (B) Alloc: 平均分配
    alloc = np.ones(NUM_CLIENTS) 
    # (C) H: 随机 8~15 步
    H_vec = np.random.randint(8, 16, size=NUM_CLIENTS)
    # (D) Ready: 指定前 8 个
    ready_vec = [0, 1, 2, 3, 4, 5, 6, 7]

    print(f"策略: Cuts={cuts}\n      Ready={ready_vec}")

    # 5. 运行计算
    res = env.event_latency(
        client_cut_layers=cuts,
        H_vec=H_vec,
        ready_vec=ready_vec
    )

    print(f"\n结果详情:")
    print(f"  T_event (总时延):     {res['T_event']:.4f} s")
    