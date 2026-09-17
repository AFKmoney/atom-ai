"""Acceptance test for exact optimizer/data/RNG checkpoint resumption."""

import random

import numpy as np
import torch

from src.io.data import InfiniteDataLoader
from src.toroidal.model import ToroidalFractalIntelligence
from src.training.trainer import ToroidalTrainer


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_trainer(tmp_path) -> ToroidalTrainer:
    data = [torch.tensor([i, i + 1, i + 2, i + 3]) % 17 for i in range(6)]
    loader = InfiniteDataLoader(data, batch_size=2, seq_len=4, shuffle=True)
    model = ToroidalFractalIntelligence(
        vocab_size=17, d_model=4, n_modes=4, n_atoms_max=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return ToroidalTrainer(
        model,
        loader,
        optimizer,
        save_dir=str(tmp_path),
        log_interval=1000,
        save_interval=1000,
    )


def _step(trainer: ToroidalTrainer, iterator):
    batch = next(iterator).clone()
    logits = []

    def capture_logits(_module, _args, output):
        logits.append(output["logits"].detach().clone())

    hook = trainer.model.register_forward_hook(capture_logits)
    try:
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, n = trainer._train_batch(batch)
        assert n > 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.step += 1
    finally:
        hook.remove()

    return (
        batch,
        loss.detach().clone(),
        torch.cat(logits),
        [parameter.detach().clone() for parameter in trainer.model.parameters()],
    )


def test_interrupted_resume_matches_continuous_next_step(tmp_path):
    seed = 20260913

    _seed_all(seed)
    continuous = _make_trainer(tmp_path / "continuous")
    continuous_iterator = iter(continuous.dataloader)
    for _ in range(3):
        _step(continuous, continuous_iterator)
    continuous_next = _step(continuous, continuous_iterator)

    _seed_all(seed)
    interrupted = _make_trainer(tmp_path / "interrupted")
    interrupted_iterator = iter(interrupted.dataloader)
    for _ in range(3):
        _step(interrupted, interrupted_iterator)
    checkpoint = interrupted.save("resume.pt")

    # Model construction consumes RNG; Trainer.load must restore the checkpoint
    # RNG after construction and restore the saved data permutation/cursor.
    reloaded = _make_trainer(tmp_path / "reloaded")
    reloaded.load(checkpoint)
    reloaded_next = _step(reloaded, iter(reloaded.dataloader))

    torch.testing.assert_close(continuous_next[0], reloaded_next[0], rtol=0, atol=0)
    torch.testing.assert_close(continuous_next[1], reloaded_next[1], rtol=0, atol=0)
    torch.testing.assert_close(continuous_next[2], reloaded_next[2], rtol=0, atol=0)
    assert len(continuous_next[3]) == len(reloaded_next[3])
    for continuous_parameter, reloaded_parameter in zip(
        continuous_next[3], reloaded_next[3]
    ):
        torch.testing.assert_close(
            continuous_parameter, reloaded_parameter, rtol=0, atol=0
        )


def test_model_only_checkpoint_remains_loadable(tmp_path):
    _seed_all(7)
    source = _make_trainer(tmp_path / "source")
    model_only = tmp_path / "model_only.pt"
    source.model.save(str(model_only))

    target = _make_trainer(tmp_path / "target")
    target.load(str(model_only))
    assert target.step == 0
    assert target.epoch == 0
    assert len(target.model.atoms) == len(source.model.atoms)
