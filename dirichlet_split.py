# dirichlet_split.py
from __future__ import annotations
from typing import List
import numpy as np


def _get_labels(dataset) -> np.ndarray:
    # torch.utils.data.Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        idx = np.array(dataset.indices, dtype=int)
        return _get_labels(base)[idx]

    # torchvision datasets
    for attr in ["targets", "labels", "y"]:
        if hasattr(dataset, attr):
            lab = getattr(dataset, attr)
            return np.asarray(lab)

    # fallback: try indexing (slow, but okay for small smoke tests)
    n = len(dataset)
    labs = np.empty(n, dtype=int)
    for i in range(n):
        _, y = dataset[i]
        labs[i] = int(y)
    return labs


def split_clients_dirichlet(
    dataset,
    num_clients: int,
    beta: float,
    seed: int = 0,
    min_size: int = 10,
    balance: bool = False,
    max_retry: int = 50,
) -> List[List[int]]:
    """
    Dirichlet non-IID split (common in FL literature).

    Returns:
        client_splits: List[List[int]]
            client_splits[cid] = list of sample indices for client cid

    约定：
    - 整个工程统一使用 List[List[int]]
    - 同一个 run 内，不同 scheme 必须复用同一份 client_splits
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if beta <= 0:
        raise ValueError("beta must be > 0")

    y = _get_labels(dataset)
    classes = np.unique(y)
    K = len(classes)

    rng = np.random.default_rng(seed)

    cls_indices = [np.where(y == c)[0] for c in classes]
    for k in range(K):
        rng.shuffle(cls_indices[k])

    for _ in range(max_retry):
        client_indices = [[] for _ in range(num_clients)]

        # draw proportions per class
        for k in range(K):
            idx_k = cls_indices[k]
            if len(idx_k) == 0:
                continue

            props = rng.dirichlet(alpha=np.ones(num_clients) * beta)

            if balance:
                props = np.minimum(props, np.median(props) * 2)
                props = props / props.sum()

            cuts = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
            split = np.split(idx_k, cuts)

            for cid, part in enumerate(split):
                client_indices[cid].extend(part.tolist())

        sizes = [len(ci) for ci in client_indices]
        if min(sizes) >= min_size:
            out: List[List[int]] = []
            for cid in range(num_clients):
                arr = np.array(client_indices[cid], dtype=int)
                rng.shuffle(arr)
                out.append(arr.tolist())
            return out

    raise RuntimeError(
        f"Failed to generate Dirichlet split with min_size={min_size} after {max_retry} retries. "
        f"Try larger beta or smaller min_size."
    )


# Same client_splits must be reused by all schemes within one run.
# This is a run-level exogenous condition.

def summarize_client_splits(client_splits: List[List[int]]):
    return {
        "num_clients": len(client_splits),
        "sizes": [len(v) for v in client_splits],
        "heads": [v[: min(8, len(v))] for v in client_splits],
    }


def hash_client_splits(client_splits: List[List[int]]):
    from repro import hash_list_of_indices
    return hash_list_of_indices(client_splits)


if __name__ == "__main__":
    """
    最小测试：
    1. 构造一个 toy label 数据集
    2. 做 Dirichlet split
    3. 打印每个 client 的样本数与前几个 index

    用途：
    - 验证 split 能正常生成
    - 验证返回类型是 List[List[int]]
    - 验证 summarize/hash 接口一致
    """
    labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 1, 0, 2]

    class DummyDS:
        targets = labels

        def __len__(self):
            return len(self.targets)

    splits = split_clients_dirichlet(
        dataset=DummyDS(),
        num_clients=3,
        beta=0.5,
        min_size=2,
        seed=2024,
    )

    print("Type:", type(splits))
    print("Summary:", summarize_client_splits(splits))
    print("Hash:", hash_client_splits(splits))