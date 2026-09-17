"""Continuous trainer for Toroidal Fractal Intelligence."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..toroidal.model import ToroidalFractalIntelligence
from ..evaluation.metrics import ToroidalMetrics
from ..io.data import InfiniteDataLoader


class ToroidalTrainer:
    """Train the continuous state one token transition at a time."""

    def __init__(self, model: ToroidalFractalIntelligence,
                 dataloader: torch.utils.data.DataLoader | InfiniteDataLoader,
                 optimizer: torch.optim.Optimizer, save_dir: str = "./checkpoints",
                 log_interval: int = 100, save_interval: int = 1000) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.metrics = ToroidalMetrics()
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.training_log: list[dict] = []

    @staticmethod
    def _tokens(batch):
        if isinstance(batch, dict):
            tokens = batch["input_ids"]
        else:
            tokens = batch
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim > 2:
            tokens = tokens.squeeze(-1)
        return tokens.long()

    def _train_batch(self, batch) -> tuple[torch.Tensor, int]:
        """Consume a batch as a continuous token stream and return mean loss."""
        tokens = self._tokens(batch)
        if tokens.shape[1] < 2:
            return tokens.new_zeros((), dtype=torch.float32), 0

        # A continuous architecture must see transitions in order. Do not flatten
        # a sequence into a fake batch and do not make one prediction serve all
        # unrelated targets.
        losses = []
        for row in tokens:
            for t in range(tokens.shape[1] - 1):
                output = self.model(row[t:t + 1])
                logits = output["logits"]
                target = row[t + 1:t + 2]
                losses.append(nn.functional.cross_entropy(logits, target))

        if not losses:
            return tokens.new_zeros((), dtype=torch.float32), 0
        return torch.stack(losses).mean(), len(losses)

    def _log_step(self, loss: float) -> dict:
        metric = {
            "step": self.step,
            "loss": loss,
            "perplexity": float(torch.exp(torch.tensor(loss))),
            "n_atoms": len(self.model.atoms),
            "n_aggregates": 0,
            "n_abstractions": 0,
            "consolidation_count": self.model.consolidation_count,
        }
        self.training_log.append(metric)
        self.metrics.log_metrics(
            perplexity=metric["perplexity"],
            n_atoms=metric["n_atoms"],
            n_aggregates=metric["n_aggregates"],
            n_abstractions=metric["n_abstractions"],
            consolidation_count=metric["consolidation_count"],
            loss=loss,
            n_parameters=sum(p.numel() for p in self.model.parameters()),
        )
        return metric

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, n = self._train_batch(batch)
            if n == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1
            total_loss += loss.item() * n
            total_tokens += n
            if self.step % self.log_interval == 0:
                m = self._log_step(loss.item())
                print(f"Step {self.step}: loss={loss.item():.4f}, perplexity={m['perplexity']:.2f}, atoms={m['n_atoms']}")
            if self.step % self.save_interval == 0:
                self.save()
        self.epoch += 1
        mean = total_loss / max(total_tokens, 1)
        return {"loss": mean, "perplexity": float(torch.exp(torch.tensor(mean))), "step": self.step, "epoch": self.epoch}

    def train(self, max_steps: int = 10000, validation_interval: int = 1000) -> dict:
        print(f"Starting training for {max_steps} steps...")
        start = time.time()
        last_loss = float("inf")
        iterator = iter(self.dataloader)
        while self.step < max_steps:
            batch = next(iterator)
            self.optimizer.zero_grad(set_to_none=True)
            loss, n = self._train_batch(batch)
            if n == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.step += 1
            last_loss = loss.item()
            if self.step % self.log_interval == 0:
                m = self._log_step(last_loss)
                print(f"Step {self.step}/{max_steps}: loss={last_loss:.4f}, perplexity={m['perplexity']:.2f}, atoms={m['n_atoms']}")
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % validation_interval == 0:
                val = self.validate()
                if val["loss"] < self.best_loss:
                    self.best_loss = val["loss"]
                    self.save("best_model.pt")
        self.save("final_model.pt")
        return {"final_loss": last_loss, "best_loss": self.best_loss, "total_steps": self.step, "total_time": time.time() - start}

    @torch.no_grad()
    def validate(self) -> dict:
        # Validation uses the same sequential transition semantics as training.
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.dataloader):
            if i >= 10:
                break
            tokens = self._tokens(batch)
            for row in tokens:
                for t in range(tokens.shape[1] - 1):
                    out = self.model(row[t:t + 1])
                    losses.append(nn.functional.cross_entropy(out["logits"], row[t + 1:t + 2]).item())
        loss = sum(losses) / max(len(losses), 1)
        return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "n_atoms": len(self.model.atoms)}

    @staticmethod
    def _get_rng_state() -> dict:
        """Capture every RNG used by the training and input pipelines."""
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _set_rng_state(state: dict) -> None:
        """Restore RNG state without advancing any generator."""
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _training_state(self) -> dict:
        """Return state needed to continue the next optimizer step exactly."""
        state = {
            "optimizer": self.optimizer.state_dict(),
            "trainer": {
                "step": self.step,
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "training_log": self.training_log,
                "model_training": self.model.training,
            },
            "rng": self._get_rng_state(),
        }
        if hasattr(self.dataloader, "state_dict"):
            state["dataloader"] = self.dataloader.state_dict()
        return state

    def save(self, filename: Optional[str] = None) -> str:
        filename = filename or f"checkpoint_step_{self.step}.pt"
        path = self.save_dir / filename
        self.model.save(str(path), extra_state=self._training_state())
        with open(self.save_dir / f"training_log_step_{self.step}.json", "w") as f:
            json.dump(self.training_log[-100:], f, indent=2)
        print(f"Saved checkpoint to {path}")
        return str(path)

    def load(self, path: str) -> None:
        training = self.model.load(path)
        if training is not None:
            optimizer_state = training.get("optimizer")
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

            trainer_state = training.get("trainer", {})
            self.step = int(trainer_state.get("step", self.step))
            self.epoch = int(trainer_state.get("epoch", self.epoch))
            self.best_loss = float(trainer_state.get("best_loss", self.best_loss))
            self.training_log = trainer_state.get("training_log", self.training_log)
            if "model_training" in trainer_state:
                self.model.train(bool(trainer_state["model_training"]))

            dataloader_state = training.get("dataloader")
            if dataloader_state is not None and hasattr(self.dataloader, "load_state_dict"):
                self.dataloader.load_state_dict(dataloader_state)

            rng_state = training.get("rng")
            if rng_state is not None:
                self._set_rng_state(rng_state)
        print(f"Loaded checkpoint from {path}")
