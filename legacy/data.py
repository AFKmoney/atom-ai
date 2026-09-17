"""
Data loading utilities for Toroidal Fractal Intelligence.

Supports:
- HuggingFace datasets (Wikipedia, Wikitext, etc.)
- Custom text datasets
- Infinite streaming for continuous learning
- Memory-mapped datasets for large-scale training
"""

from __future__ import annotations

import torch
from typing import Iterator, List, Optional, Union
from pathlib import Path
import json

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


def load_wikitext(
    subset: str = "wikitext-2-raw-v1",
    max_tokens: Optional[int] = None,
    seq_len: int = 128,
) -> InfiniteDataLoader:
    """
    Load Wikipedia text from HuggingFace datasets.

    Parameters
    ----------
    subset : str
        Dataset subset (e.g., "wikitext-2-raw-v1")
    max_tokens : int or None
        Maximum tokens to load (None = all)
    seq_len : int
        Sequence length for training

    Returns
    -------
    loader : InfiniteDataLoader
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    print(f"Loading {subset}...")
    dataset = load_dataset("wikitext", subset, split="train")

    # Combine all text
    text = "\n".join(dataset["text"])
    print(f"Total text length: {len(text)} characters")

    if max_tokens:
        # Tokenize and limit
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)[:max_tokens]
        print(f"Limited to {max_tokens} tokens")
    else:
        # Tokenize all
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(text)
        print(f"Total tokens: {len(tokens)}")

    # Create sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        seq = torch.tensor(tokens[i:i + seq_len])
        sequences.append(seq)

    print(f"Created {len(sequences)} sequences of length {seq_len}")

    return InfiniteDataLoader(sequences, batch_size=32, seq_len=seq_len)


def load_custom_text(
    text: str,
    seq_len: int = 128,
    max_tokens: Optional[int] = None,
) -> InfiniteDataLoader:
    """
    Load custom text data.

    Parameters
    ----------
    text : str
        Raw text data
    seq_len : int
        Sequence length
    max_tokens : int or None
        Maximum tokens (None = all)

    Returns
    -------
    loader : InfiniteDataLoader
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if max_tokens:
        tokens = tokenizer.encode(text)[:max_tokens]
    else:
        tokens = tokenizer.encode(text)

    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        seq = torch.tensor(tokens[i:i + seq_len])
        sequences.append(seq)

    return InfiniteDataLoader(sequences, batch_size=32, seq_len=seq_len)


def load_from_file(
    filepath: str,
    seq_len: int = 128,
    max_tokens: Optional[int] = None,
) -> InfiniteDataLoader:
    """
    Load text from file.

    Parameters
    ----------
    filepath : str
        Path to text file
    seq_len : int
        Sequence length
    max_tokens : int or None
        Maximum tokens

    Returns
    -------
    loader : InfiniteDataLoader
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return load_custom_text(text, seq_len=seq_len, max_tokens=max_tokens)


def create_memory_mapped_dataset(
    data_path: str,
    vocab_size: int = 50257,
    seq_len: int = 128,
    cache_dir: str = "./cache/datasets"
) -> InfiniteDataLoader:
    """
    Create memory-mapped dataset for large-scale training.

    Parameters
    ----------
    data_path : str
        Path to directory with tokenized data
    vocab_size : int
        Vocabulary size
    seq_len : int
        Sequence length
    cache_dir : str
        Cache directory for mmap files

    Returns
    -------
    loader : InfiniteDataLoader
    """
    import numpy as np
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    data_file = cache_path / "data.mmap"
    indices_file = cache_path / "indices.bin"

    if data_file.exists() and indices_file.exists():
        # Load existing mmap
        data = np.memmap(data_file, dtype=np.uint16, mode="r")
        indices = np.memmap(indices_file, dtype=np.int64, mode="r")
        print(f"Loaded memory-mapped dataset: {len(data)} tokens, {len(indices)} sequences")
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run tokenization first.")

    # Convert to torch tensors
    sequences = [torch.from_numpy(data[i:i + seq_len].copy()) for i in indices]

    return InfiniteDataLoader(sequences, batch_size=64, seq_len=seq_len)


def tokenize_and_save(
    text: str,
    output_dir: str,
    vocab_size: int = 50257,
    seq_len: int = 128,
):
    """
    Tokenize text and save as memory-mapped dataset.

    Parameters
    ----------
    text : str
        Raw text
    output_dir : str
        Output directory
    vocab_size : int
        Vocabulary size
    seq_len : int
        Sequence length
    """
    import numpy as np
    from transformers import AutoTokenizer
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)

    # Create sequences
    sequences = []
    for i in range(0, len(tokens) - seq_len, seq_len // 2):
        sequences.append(tokens[i:i + seq_len])

    # Save as mmap
    data = np.array(sequences, dtype=np.uint16)
    data_file = output_path / "data.mmap"
    data_file_tmp = data_file.with_suffix(".tmp")

    mm = np.memmap(data_file_tmp, dtype=np.uint16, mode="w+", shape=data.shape)
    mm[:] = data[:]
    mm.flush()
    del mm

    data_file_tmp.rename(data_file)
    print(f"Saved {len(sequences)} sequences to {data_file}")


if __name__ == "__main__":
    # Test data loading
    print("Testing data loaders...")

    # Test Wikitext
    print("\n1. Loading Wikitext...")
    try:
        loader = load_wikitext(max_tokens=10000)
        print(f"   Created loader with {len(loader.data)} sequences")
        batch = next(iter(loader))
        print(f"   Batch shape: {batch.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test custom text
    print("\n2. Loading custom text...")
    text = "The quick brown fox jumps over the lazy dog. " * 100
    loader = load_custom_text(text, seq_len=64)
    print(f"   Created loader with {len(loader.data)} sequences")
    batch = next(iter(loader))
    print(f"   Batch shape: {batch.shape}")

    print("\nData loading tests complete!")
