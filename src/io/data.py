"""InfiniteDataLoader for resume/order tests.

Atom-native training reads raw UTF-8 via ``Atomizer``.
Legacy GPT-2 / AutoTokenizer loaders are intentionally excluded from this repo.
"""

from __future__ import annotations

import torch
from typing import Iterator, List

class InfiniteDataLoader:
    """
    Infinite data loader for continuous learning.

    Wraps any iterable dataset and provides infinite batching.
    Supports shuffling, sliding windows, and custom sampling.
    """

    def __init__(
        self,
        data: List[torch.Tensor],
        batch_size: int = 32,
        seq_len: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Parameters
        ----------
        data : List[torch.Tensor]
            List of token sequences
        batch_size : int
            Number of sequences per batch
        seq_len : int
            Length of each sequence
        shuffle : bool
            Shuffle data each epoch
        drop_last : bool
            Drop incomplete batches
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.step = 0
        self._indices: torch.Tensor | None = None
        self._cursor = 0

    def _prepare_epoch(self) -> None:
        self._indices = torch.randperm(len(self.data)) if self.shuffle else torch.arange(len(self.data))
        self._cursor = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield infinite batches while keeping a restorable cursor."""
        while True:
            if self._indices is None or self._cursor + self.batch_size > len(self._indices):
                if self._indices is not None:
                    self.epoch += 1
                self._prepare_epoch()

            batch_indices = self._indices[self._cursor:self._cursor + self.batch_size]
            self._cursor += self.batch_size
            batch = torch.stack([self.data[idx] for idx in batch_indices])
            self.step += 1
            yield batch

    def state_dict(self) -> dict:
        """Return the exact data-order and cursor state for checkpoint resume."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "cursor": self._cursor,
            "indices": self._indices.clone() if self._indices is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore data order and cursor before creating the next iterator."""
        self.epoch = int(state.get("epoch", 0))
        self.step = int(state.get("step", 0))
        self._cursor = int(state.get("cursor", 0))
        indices = state.get("indices")
        self._indices = indices.clone().long() if indices is not None else None

    def __len__(self) -> int:
        """Number of full batches per epoch."""
        n = len(self.data) // self.batch_size
        return n if self.drop_last else n + 1


# Legacy GPT-2 / AutoTokenizer loaders are excluded from atom-ai.
# Prefer Atomizer + tools/run_atom_native.py for training.
