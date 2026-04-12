# batch_stream.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


@dataclass
class BatchCursorState:
    epoch: int = 0
    ptr: int = 0


class ClientBatchStream:
    """
    One persistent shuffled index stream per client.

    Properties:
      - samples are consumed almost exhaustively before repetition
      - order is shuffled per local epoch
      - fixed batch size
      - if remainder < batch_size, wrap into next permutation to fill the batch
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        batch_size: int,
        seed: int,
        device: Optional[torch.device] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(indices) == 0:
            raise ValueError("ClientBatchStream requires non-empty indices")

        self.dataset = dataset
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(int(seed))
        self.device = device

        self.state = BatchCursorState(epoch=0, ptr=0)
        self._perm = self.rng.permutation(self.indices)

    def __len__(self) -> int:
        return int(len(self.indices))

    def _reshuffle(self) -> None:
        self._perm = self.rng.permutation(self.indices)
        self.state.epoch += 1
        self.state.ptr = 0

    def next_indices(self) -> List[int]:
        B = self.batch_size
        n = len(self._perm)
        ptr = self.state.ptr

        if ptr + B <= n:
            out = self._perm[ptr:ptr + B]
            self.state.ptr = ptr + B
            if self.state.ptr == n:
                # next call will start a fresh permutation
                self._reshuffle()
            return out.tolist()

        # wrap-around case: use tail of current permutation + head of next permutation
        tail = self._perm[ptr:].tolist()
        need = B - len(tail)

        self._reshuffle()
        head = self._perm[:need].tolist()
        self.state.ptr = need
        return tail + head

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_indices = self.next_indices()
        items = [self.dataset[i] for i in batch_indices]
        batch = default_collate(items)

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError(f"Unexpected batch format: {type(batch)}")

        x, y = batch[0], batch[1]
        return x, y