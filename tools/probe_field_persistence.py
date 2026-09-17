#!/usr/bin/env python3
"""Read-only scientific probe: does the toroidal FIELD carry prompt structure?

Thesis (user): training still looks like a small LM (packet CE). The surface
can learn byte bigrams while the field is ignored. This probe judges the
field / consolidation / alpha state — not only CE or generation quality.

Constraints:
  - Read-only: load checkpoint, eval mode, no optimizer, no weight updates.
  - Does not stop or modify any running training process.
  - No attention, no GPT-2, no MERGE/SPLIT implementation.

Usage:
  PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py
  PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \\
      --checkpoint checkpoints/atom_native_chat_talk/atom_native.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.atom_native import DEFAULT_ENERGY_DECAY_BOUNDS, AtomNativeModel
from src.io.atomizer import AtomPacket

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPTS = [
    "Bonjour",
    "Qui es-tu ?",
    "Il était une fois",
    "Utilisateur: Bonjour\nAssistant:",  # short dialogue turn (already primed)
]

PREFERRED_CKPTS = [
    REPO_ROOT / "checkpoints/atom_native_stream_dialogue/atom_native.pt",
    REPO_ROOT / "checkpoints/atom_native_chat_talk/atom_native.pt",
]


def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"checkpoint missing or empty: {path}")
        return path
    for candidate in PREFERRED_CKPTS:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        "no usable checkpoint; tried:\n  " + "\n  ".join(str(p) for p in PREFERRED_CKPTS)
    )


def load_model(checkpoint: Path, device: str = "cpu") -> tuple[AtomNativeModel, dict]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = blob.get("config") or {}
    bounds = config.get("energy_decay_bounds")
    energy_decay_bounds = tuple(bounds) if bounds is not None else DEFAULT_ENERGY_DECAY_BOUNDS
    field_max_rms = config.get("field_max_rms")
    if field_max_rms is None:
        field_max_rms = 3.0
    model = AtomNativeModel(
        d_model=int(config.get("d_model", 64)),
        n_modes=int(config.get("n_modes", 64)),
        n_atoms_max=int(config.get("n_atoms_max", 512)),
        max_payload_bytes=int(config.get("max_payload_bytes", 16)),
        field_max_rms=field_max_rms,
        energy_decay_bounds=energy_decay_bounds,
    )
    training = model.load(checkpoint) or {}
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {
        "config": config,
        "training_meta": {
            "step": training.get("step"),
            "run_steps": training.get("run_steps"),
            "seed": training.get("seed"),
            "episode_reset": training.get("episode_reset"),
            "atom_flush_every": training.get("atom_flush_every"),
            "dynamics_on_load": training.get("dynamics_on_load"),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def role_prime(prompt: str) -> str:
    if "Assistant:" in prompt or "Utilisateur:" in prompt:
        return prompt
    return f"Utilisateur: {prompt}\nAssistant:"


def field_stats(model: AtomNativeModel) -> dict[str, float]:
    alpha = model.core.state.alpha.detach()
    rms = float(torch.sqrt(alpha.pow(2).mean()).item())
    norm = float(alpha.norm().item())
    mean_abs = float(alpha.abs().mean().item())
    max_abs = float(alpha.abs().max().item())
    persist = model.core.consolidation.persistence.detach()
    persist_str = float(model.core.consolidation.persistence_strength.detach().item())
    return {
        "field_rms": rms,
        "field_norm": norm,
        "field_mean_abs": mean_abs,
        "field_max_abs": max_abs,
        "alpha_numel": int(alpha.numel()),
        "persistence_norm": float(persist.norm().item()),
        "persistence_mean_abs": float(persist.abs().mean().item()),
        "persistence_strength": persist_str,
        "n_atoms": int(len(model.core.atoms)),
        "consolidation_count": int(model.core.consolidation_count),
        "abstraction_count": int(model.core.abstraction_count),
        "t": float(model.core.state.t.detach().item()),
    }


def flatten_logits(surface: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [surface["byte_logits"].reshape(-1), surface["length_logits"].reshape(-1)]
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() != b.numel():
        n = min(a.numel(), b.numel())
        a, b = a[:n], b[:n]
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    return float(torch.norm(a[:n] - b[:n]).item())


@torch.no_grad()
def prime_prompt(
    model: AtomNativeModel,
    prompt: str,
    *,
    use_role_prime: bool = True,
) -> dict[str, Any]:
    """Encode prompt and inject all packets into the field (chat priming path).

    Returns the last forward output plus collected packet feature vectors.
    """
    model.reset_state(reset_atomizer=True)
    primed = role_prime(prompt) if use_role_prime else prompt
    packets = model.atomizer.encode(primed, reset=True)
    if not packets:
        raise ValueError(f"prompt produced no packets: {prompt!r}")

    last_out: dict | None = None
    packet_features: list[torch.Tensor] = []
    packet_payloads: list[str] = []
    for packet in packets:
        last_out = model.forward_packet(packet)
        packet_features.append(packet.features.detach().cpu().float().clone())
        packet_payloads.append(packet.payload.decode("utf-8", errors="replace"))

    assert last_out is not None
    alpha = model.core.state.alpha.detach().cpu().clone()
    output_state = model._output_state(
        last_out["field"].detach(),
        last_out.get("persistent_state"),
        last_out.get("abstractions") or [],
    ).detach().cpu().clone()
    surface_flat = flatten_logits(last_out["surface"]).cpu().clone()

    return {
        "prompt": prompt,
        "primed_text": primed,
        "n_prompt_packets": len(packets),
        "packet_payloads": packet_payloads,
        "packet_features": packet_features,
        "stats_after_prompt": field_stats(model),
        "alpha": alpha,
        "output_state": output_state,
        "surface_logits_flat": surface_flat,
        "persistence": model.core.consolidation.persistence.detach().cpu().clone(),
        "last_surface": {
            "byte_logits": last_out["surface"]["byte_logits"].detach().cpu().clone(),
            "length_logits": last_out["surface"]["length_logits"].detach().cpu().clone(),
        },
    }


@torch.no_grad()
def step_generate(
    model: AtomNativeModel,
    n_packets: int = 20,
    *,
    temperature: float = 0.7,
    top_k: int = 8,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Generate n packets after priming (mutates model field state).

    Uses the last atomizer context: caller must have primed already.
    We re-encode nothing; we need a current packet. After priming, the
    atomizer buffer is empty — use the last injected payload via a
    synthetic current from the last prompt packet features path:
    generate_packets keeps ``current`` as the last prompt packet. Here we
    reconstruct that by reading energy_history length / atoms if needed.

    Honest approach: re-prime is done by caller; we take the last packet
    from a stash passed in. So this function requires ``current`` packet.
    """
    raise NotImplementedError("use generate_after_prime")


@torch.no_grad()
def generate_after_prime(
    model: AtomNativeModel,
    current: AtomPacket,
    n_packets: int = 20,
    *,
    temperature: float = 0.7,
    top_k: int = 8,
    deterministic: bool = True,
) -> dict[str, Any]:
    generated = bytearray()
    for _ in range(n_packets):
        payload = model.predict_packet(
            current,
            temperature=temperature,
            top_k=top_k,
            deterministic=deterministic,
            prefer_printable=True,
        )
        if not payload:
            break
        generated.extend(payload)
        current = model.atomizer.packet_from_payload(payload)
    return {
        "n_generated_packets": n_packets,
        "generated_bytes": len(generated),
        "generated_preview": bytes(generated[:120]).decode("utf-8", errors="replace"),
        "stats_after_gen": field_stats(model),
    }


@torch.no_grad()
def run_prompt_trial(
    model: AtomNativeModel,
    prompt: str,
    n_gen: int = 20,
    *,
    use_role_prime: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Prime full prompt into field, snapshot, then generate n_gen packets."""
    primed = role_prime(prompt) if use_role_prime else prompt

    model.reset_state(reset_atomizer=True)
    torch.manual_seed(seed)
    packets = model.atomizer.encode(primed, reset=True)
    if not packets:
        raise ValueError(f"prompt produced no packets: {prompt!r}")

    last_out = None
    packet_features: list[torch.Tensor] = []
    packet_payloads: list[str] = []
    for packet in packets:
        last_out = model.forward_packet(packet)
        packet_features.append(packet.features.detach().cpu().float().clone())
        packet_payloads.append(packet.payload.decode("utf-8", errors="replace"))
    assert last_out is not None

    stats_after_prompt = field_stats(model)
    alpha_after = model.core.state.alpha.detach().cpu().clone()
    output_state = model._output_state(
        last_out["field"].detach(),
        last_out.get("persistent_state"),
        last_out.get("abstractions") or [],
    ).detach().cpu().clone()
    surface_flat = flatten_logits(last_out["surface"]).cpu().clone()
    persistence = model.core.consolidation.persistence.detach().cpu().clone()
    surface_byte_logits_norm = float(last_out["surface"]["byte_logits"].norm().item())

    # Generation path mirrors generate_packets (forward all but last, then predict).
    model.reset_state(reset_atomizer=True)
    torch.manual_seed(seed)
    packets = model.atomizer.encode(primed, reset=True)
    for packet in packets[:-1]:
        model.forward_packet(packet)
    current = packets[-1]
    generated = bytearray()
    for _ in range(n_gen):
        payload = model.predict_packet(
            current,
            temperature=0.7,
            top_k=8,
            deterministic=True,
            prefer_printable=True,
        )
        if not payload:
            break
        generated.extend(payload)
        current = model.atomizer.packet_from_payload(payload)
    stats_after_gen = field_stats(model)

    return {
        "prompt": prompt,
        "primed_text": primed,
        "n_prompt_packets": len(packets),
        "packet_payloads": packet_payloads,
        "packet_features": packet_features,
        "stats_after_prompt": stats_after_prompt,
        "stats_after_gen": stats_after_gen,
        "alpha": alpha_after,
        "output_state": output_state,
        "surface_logits_flat": surface_flat,
        "persistence": persistence,
        "generated_preview": bytes(generated[:160]).decode("utf-8", errors="replace"),
        "generated_bytes": len(generated),
        "n_gen_requested": n_gen,
        "surface_byte_logits_norm": surface_byte_logits_norm,
    }



def pairwise_sensitivity(trials: list[dict]) -> dict[str, Any]:
    names = [t["prompt"] for t in trials]
    n = len(trials)
    cos_alpha = [[0.0] * n for _ in range(n)]
    cos_output = [[0.0] * n for _ in range(n)]
    cos_logits = [[0.0] * n for _ in range(n)]
    cos_persist = [[0.0] * n for _ in range(n)]
    dist_alpha = [[0.0] * n for _ in range(n)]
    off_diag_alpha: list[float] = []
    off_diag_output: list[float] = []
    off_diag_logits: list[float] = []
    off_diag_persist: list[float] = []

    for i in range(n):
        for j in range(n):
            cos_alpha[i][j] = cosine(trials[i]["alpha"], trials[j]["alpha"])
            cos_output[i][j] = cosine(trials[i]["output_state"], trials[j]["output_state"])
            cos_logits[i][j] = cosine(
                trials[i]["surface_logits_flat"], trials[j]["surface_logits_flat"]
            )
            cos_persist[i][j] = cosine(trials[i]["persistence"], trials[j]["persistence"])
            dist_alpha[i][j] = l2_distance(trials[i]["alpha"], trials[j]["alpha"])
            if i < j:
                off_diag_alpha.append(cos_alpha[i][j])
                off_diag_output.append(cos_output[i][j])
                off_diag_logits.append(cos_logits[i][j])
                off_diag_persist.append(cos_persist[i][j])

    def _mean(xs: list[float]) -> float:
        xs = [x for x in xs if x == x]  # drop nan
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "prompts": names,
        "cosine_alpha": cos_alpha,
        "cosine_output_state": cos_output,
        "cosine_surface_logits": cos_logits,
        "cosine_persistence": cos_persist,
        "l2_alpha": dist_alpha,
        "mean_offdiag_cosine_alpha": _mean(off_diag_alpha),
        "mean_offdiag_cosine_output_state": _mean(off_diag_output),
        "mean_offdiag_cosine_surface_logits": _mean(off_diag_logits),
        "mean_offdiag_cosine_persistence": _mean(off_diag_persist),
        "interpretation": (
            "If inter-prompt cosine of surface logits / output_state ≈ 1.0, "
            "the surface largely ignores prompt-conditioned field differences. "
            "Meaningfully < 1 suggests prompt-sensitive state."
        ),
    }


def reconstruction_probe(
    model: AtomNativeModel,
    trials: list[dict],
    *,
    last_k: int = 4,
) -> dict[str, Any]:
    """Best-effort field→past-packet recovery WITHOUT using AR surface loop.

    Architecture reality:
      - No MERGE operator, no field→past-packet decoder.
      - Consolidation is a single persistence vector (EMA of selected alpha).
      - Atom collection stores compiled (r,phi,...) for atoms seen this episode.

    Honest tests:
      (a) Do alpha / consolidation / atom-collection still differ across prompts?
      (b) Linear probe: least-squares map from flattened field features → mean
          of last-k packet feature vectors (fit on these few prompts only —
          diagnostic, not a claim of generalization).
      (c) Nearest-neighbor: match each prompt's last-k feature mean against
          stored atom.r collection (if available) or against persistence.
    """
    notes = [
        "No MERGE/SPLIT in this architecture yet.",
        "No dedicated field→past-packet decoder exists.",
        "Consolidation is a single EMA persistence vector, not a packet stack.",
        "Linear probe below is few-shot diagnostic on the probe prompts only.",
    ]

    # (a) distances already in sensitivity; summarize field-side separation
    alphas = torch.stack([t["alpha"].reshape(-1) for t in trials])
    persists = torch.stack([t["persistence"].reshape(-1) for t in trials])
    pairwise_alpha_l2 = []
    pairwise_persist_l2 = []
    for i in range(len(trials)):
        for j in range(i + 1, len(trials)):
            pairwise_alpha_l2.append(float(torch.norm(alphas[i] - alphas[j]).item()))
            pairwise_persist_l2.append(float(torch.norm(persists[i] - persists[j]).item()))

    field_differs = (
        max(pairwise_alpha_l2) > 1e-4 if pairwise_alpha_l2 else False
    )
    persist_differs = (
        max(pairwise_persist_l2) > 1e-6 if pairwise_persist_l2 else False
    )

    # Build targets: mean of last-k packet features per prompt
    targets = []
    fields = []
    for t in trials:
        feats = t["packet_features"]
        k = min(last_k, len(feats))
        mean_feat = torch.stack(feats[-k:]).mean(dim=0)
        targets.append(mean_feat)
        # Field feature for probe: flatten alpha + persistence + output_state
        fields.append(
            torch.cat(
                [
                    t["alpha"].reshape(-1),
                    t["persistence"].reshape(-1),
                    t["output_state"].reshape(-1),
                ]
            )
        )
    X = torch.stack(fields)  # [P, D]
    Y = torch.stack(targets)  # [P, F]
    P, D = X.shape
    Fdim = Y.shape[1]

    linear_result: dict[str, Any]
    if P < 2:
        linear_result = {"status": "skipped", "reason": "need >=2 prompts"}
    else:
        # Ridge least squares: W = (X'X + λI)^{-1} X'Y  → predict Yhat = X W
        # With P << D this is underdetermined; use X X' form (dual ridge).
        lam = 1e-2
        # Center
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        # Dual: W_eff such that pred = Xc @ Xc.T @ (Xc Xc.T + λI)^{-1} @ Yc
        G = Xc @ Xc.T
        A = G + lam * torch.eye(P)
        try:
            coeff = torch.linalg.solve(A, Yc)  # [P, F]
            Yhat_c = G @ coeff
            Yhat = Yhat_c + Y.mean(dim=0, keepdim=True)
            # Leave-one-out style: report train MSE and cosine of preds vs targets
            mse = float(((Yhat - Y) ** 2).mean().item())
            cosines = [cosine(Yhat[i], Y[i]) for i in range(P)]
            # Identity baseline: predict global mean
            baseline_mse = float(((Y.mean(dim=0, keepdim=True).expand_as(Y) - Y) ** 2).mean().item())
            linear_result = {
                "status": "ok",
                "method": "dual_ridge_least_squares",
                "lambda": lam,
                "n_prompts": P,
                "field_dim": D,
                "feature_dim": Fdim,
                "train_mse": mse,
                "baseline_mean_mse": baseline_mse,
                "mse_improvement_vs_mean": baseline_mse - mse,
                "pred_target_cosine_per_prompt": cosines,
                "mean_pred_target_cosine": float(sum(cosines) / len(cosines)),
                "note": (
                    "In-sample fit only (P prompts). Success means field features "
                    "linearly span last-k packet features on this set — not a "
                    "general decoder."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            linear_result = {"status": "failed", "error": str(exc)}

    # (c) nearest-neighbor among prompts using alpha as key, last-k feat as value
    # Self-match trivial; report whether alpha-NN retrieves correct prompt identity
    # under leave-one-out: for each i, find nearest j≠i by alpha; check if
    # that is meaningless for recovery — instead: can we recover which prompt
    # from alpha alone via 1-NN in alpha space? (identity classification)
    nn_correct = 0
    nn_total = 0
    for i in range(P):
        # leave-one-out: classify i by nearest other — always wrong by construction
        # Better test: duplicate-run stability — re-prime and match.
        pass
    # Stability NN: re-run each prompt and match to gallery
    gallery_alpha = [t["alpha"].reshape(-1) for t in trials]
    nn_hits = 0
    for i, t in enumerate(trials):
        model.reset_state(reset_atomizer=True)
        torch.manual_seed(0)
        primed = t["primed_text"]
        packets = model.atomizer.encode(primed, reset=True)
        for packet in packets:
            model.forward_packet(packet)
        query = model.core.state.alpha.detach().cpu().reshape(-1)
        dists = [float(torch.norm(query - g).item()) for g in gallery_alpha]
        pred = int(min(range(len(dists)), key=lambda j: dists[j]))
        if pred == i:
            nn_hits += 1
        nn_total += 1

    # Atom collection: after priming, do stored atom.r vectors NN-match packet features?
    # Packet features dim != d_model necessarily — compare via compiler projection.
    atom_nn: dict[str, Any] = {"status": "attempted"}
    try:
        atom_match_scores = []
        for t in trials:
            model.reset_state(reset_atomizer=True)
            torch.manual_seed(0)
            packets = model.atomizer.encode(t["primed_text"], reset=True)
            compiled_r = []
            for packet in packets:
                feats = packet.features.to(model.core.state.alpha.device)
                atom, _, _ = model.compiler(feats, atom_count=len(compiled_r))
                compiled_r.append(atom.r.detach().cpu().float().reshape(-1))
                model.forward_packet(packet)
            if not compiled_r or len(model.core.atoms) == 0:
                atom_match_scores.append(float("nan"))
                continue
            # Stored atoms.r vs compiled r of last-k packets: cosine mean
            stored = model.core.atoms.r.detach().cpu().float()
            if stored.ndim == 1:
                stored = stored.unsqueeze(0)
            k = min(last_k, len(compiled_r), stored.shape[0])
            # Compare last-k compiled to last-k stored rows
            sims = []
            for j in range(1, k + 1):
                sims.append(cosine(compiled_r[-j], stored[-j].reshape(-1)))
            atom_match_scores.append(float(sum(sims) / len(sims)))
        atom_nn = {
            "status": "ok",
            "mean_last_k_compiled_vs_stored_r_cosine": float(
                sum(x for x in atom_match_scores if x == x) / max(1, sum(1 for x in atom_match_scores if x == x))
            ),
            "per_prompt": atom_match_scores,
            "note": (
                "Atom collection stores detached compiled atoms from this episode; "
                "high cosine is expected (same episode write). This is NOT recovery "
                "from field alone — it shows structural memory buffer retention."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        atom_nn = {"status": "failed", "error": str(exc)}

    # Verdict pieces
    linear_ok = (
        linear_result.get("status") == "ok"
        and float(linear_result.get("mse_improvement_vs_mean") or 0) > 1e-6
        and float(linear_result.get("mean_pred_target_cosine") or 0) > 0.5
    )
    reconstruction_succeeds = bool(linear_ok and field_differs)

    return {
        "last_k": last_k,
        "notes": notes,
        "not_possible_yet": [
            "MERGE of packets into durable field structures",
            "SPLIT / dual-clock span control",
            "Dedicated field → past-packet decoder",
            "Retrieving arbitrary past payloads from alpha alone as text",
        ],
        "field_differs_across_prompts": field_differs,
        "persistence_differs_across_prompts": persist_differs,
        "pairwise_alpha_l2": pairwise_alpha_l2,
        "pairwise_persistence_l2": pairwise_persist_l2,
        "mean_pairwise_alpha_l2": float(sum(pairwise_alpha_l2) / len(pairwise_alpha_l2))
        if pairwise_alpha_l2
        else float("nan"),
        "linear_probe": linear_result,
        "alpha_reprime_1nn_accuracy": nn_hits / max(nn_total, 1),
        "atom_collection_probe": atom_nn,
        "reconstruction_succeeds": reconstruction_succeeds,
        "reconstruction_summary": (
            "partial/diagnostic linear map from field features to last-k packet "
            "features works in-sample"
            if reconstruction_succeeds
            else "no honest field→past-packet reconstruction yet (surface AR not counted)"
        ),
    }


def verdict_from(report: dict) -> dict[str, Any]:
    sens = report["prompt_sensitivity"]
    recon = report["reconstruction"]
    mean_logit_cos = sens["mean_offdiag_cosine_surface_logits"]
    mean_alpha_cos = sens["mean_offdiag_cosine_alpha"]
    rms_rows = report["per_prompt"]

    rms_prompt = [r["stats_after_prompt"]["field_rms"] for r in rms_rows]
    rms_gen = [r["stats_after_gen"]["field_rms"] for r in rms_rows]

    # Heuristic verdict
    surface_collapses = mean_logit_cos > 0.995
    field_collapses = mean_alpha_cos > 0.995 and recon["mean_pairwise_alpha_l2"] < 1e-3
    carries_structure = (
        recon["field_differs_across_prompts"]
        and mean_alpha_cos < 0.99
        and (recon["reconstruction_succeeds"] or mean_logit_cos < 0.98)
    )

    if field_collapses and surface_collapses:
        label = "surface-only byte-LM (field ignored / collapsed across prompts)"
    elif recon["field_differs_across_prompts"] and surface_collapses:
        # Field has prompt structure but the surface head does not express it —
        # classic CE-down / gen-noise Transformer-regime-without-farm failure.
        label = (
            "field differs across prompts but surface logits nearly identical — "
            "field not reliably read by surface (Transformer-regime risk / "
            "surface-only byte-LM behavior at decode)"
        )
    elif carries_structure and recon["reconstruction_succeeds"]:
        label = "field is carrying structure (prompt-sensitive; diagnostic reconstruction partial)"
    elif recon["field_differs_across_prompts"]:
        label = (
            "field shows prompt sensitivity; reconstruction limited — "
            "not yet a farm of persistent matter"
        )
    else:
        label = "surface-only byte-LM (no reliable field separation)"

    return {
        "label": label,
        "mean_offdiag_cosine_surface_logits": mean_logit_cos,
        "mean_offdiag_cosine_alpha": mean_alpha_cos,
        "mean_field_rms_after_prompt": float(sum(rms_prompt) / len(rms_prompt)),
        "mean_field_rms_after_20_gen": float(sum(rms_gen) / len(rms_gen)),
        "reconstruction_succeeds": recon["reconstruction_succeeds"],
        "regime_note": (
            "CE down + gen still noise without field reconstruction = "
            "still Transformer regime without the farm."
        ),
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return None  # drop large tensors from JSON
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if not isinstance(v, torch.Tensor)}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_markdown(report: dict, path: Path) -> None:
    v = report["verdict"]
    sens = report["prompt_sensitivity"]
    lines = [
        "# FIELD_PROBE — persistence vs surface-only byte-LM",
        "",
        f"_Generated: {report['timestamp_local']}_",
        "",
        "## Thesis",
        "",
        "We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.",
        "The field can be ignored while the surface learns byte bigrams.",
        "This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.",
        "",
        "## Checkpoint",
        "",
        f"- Path: `{report['meta']['checkpoint']}`",
        f"- Bytes: {report['meta']['checkpoint_bytes']}",
        f"- Config: `{json.dumps(report['meta']['config'])}`",
        f"- Training step: `{report['meta']['training_meta'].get('step')}`",
        "",
        "## Hard numbers",
        "",
        "| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |",
        "|--------|------------------|------------------|-------|----------------------|",
    ]
    for row in report["per_prompt"]:
        a = row["stats_after_prompt"]["field_rms"]
        b = row["stats_after_gen"]["field_rms"]
        lines.append(
            f"| {row['prompt']!r} | {a:.6f} | {b:.6f} | {b - a:+.6f} | "
            f"{row['stats_after_prompt']['n_atoms']} |"
        )
    lines += [
        "",
        f"- Mean RMS after prompt: **{v['mean_field_rms_after_prompt']:.6f}**",
        f"- Mean RMS after 20 gens: **{v['mean_field_rms_after_20_gen']:.6f}**",
        "",
        "### Prompt sensitivity (off-diagonal mean cosine)",
        "",
        f"- Alpha (field): **{sens['mean_offdiag_cosine_alpha']:.6f}**",
        f"- Output state: **{sens['mean_offdiag_cosine_output_state']:.6f}**",
        f"- Surface logits: **{sens['mean_offdiag_cosine_surface_logits']:.6f}**",
        f"- Persistence: **{sens['mean_offdiag_cosine_persistence']:.6f}**",
        "",
        "Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; "
        "meaningfully < 1 ⇒ prompt-sensitive state.",
        "",
        "### Pairwise surface-logit cosine matrix",
        "",
        "```",
        json.dumps(sens["cosine_surface_logits"], indent=2),
        "```",
        "",
        "### Pairwise alpha cosine matrix",
        "",
        "```",
        json.dumps(sens["cosine_alpha"], indent=2),
        "```",
        "",
        "## Field reconstruction probe",
        "",
        f"- Field differs across prompts: **{report['reconstruction']['field_differs_across_prompts']}**",
        f"- Persistence differs: **{report['reconstruction']['persistence_differs_across_prompts']}**",
        f"- Mean pairwise alpha L2: **{report['reconstruction']['mean_pairwise_alpha_l2']:.6f}**",
        f"- Alpha re-prime 1-NN accuracy: **{report['reconstruction']['alpha_reprime_1nn_accuracy']:.3f}**",
        f"- Reconstruction succeeds (diagnostic): **{report['reconstruction']['reconstruction_succeeds']}**",
        f"- Summary: {report['reconstruction']['reconstruction_summary']}",
        "",
        "Linear probe:",
        "```json",
        json.dumps(report["reconstruction"]["linear_probe"], indent=2),
        "```",
        "",
        "Atom collection probe:",
        "```json",
        json.dumps(report["reconstruction"]["atom_collection_probe"], indent=2),
        "```",
        "",
        "### Not possible yet",
        "",
    ]
    for item in report["reconstruction"]["not_possible_yet"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Verdict",
        "",
        f"**{v['label']}**",
        "",
        v["regime_note"],
        "",
        "## Next levers (not implemented in this probe)",
        "",
        "- Real **MERGE** (packets → durable field structures)",
        "- **Dual clocks** (fast surface / slow field)",
        "- **Adaptive spans** (beyond fixed max_span_bytes)",
        "",
        "## Method notes",
        "",
        "- Read-only: `model.eval()`, all `requires_grad=False`, no optimizer.",
        "- Priming path matches chat: Atomizer encode → `forward_packet` per packet.",
        "- Generation: 20 deterministic packets after prompt (temperature 0.7 path with deterministic=True).",
        "- Did not stop or modify any running training process.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_persistence_test_doc(report: dict, path: Path) -> None:
    v = report["verdict"]
    text = f"""# PERSISTENCE_TEST — field farm vs Transformer regime

## One-liner

**CE down + gen still noise without field reconstruction = still Transformer regime without the farm.**

## Why this test exists

Packet CE can fall while the model behaves like a small byte-LM: the surface
head predicts local bigrams and the toroidal field is unused or collapsed.
Capability claims that rely on “persistent structured matter” require a probe
that asks whether the **field** still differs by prompt and whether past
packet information can be recovered from field/consolidation state — not only
whether CE improved.

## What we measured (this run)

| Metric | Value |
|--------|-------|
| Checkpoint | `{report['meta']['checkpoint']}` |
| Mean field RMS after prompt | {v['mean_field_rms_after_prompt']:.6f} |
| Mean field RMS after 20 gen | {v['mean_field_rms_after_20_gen']:.6f} |
| Mean off-diag cosine (surface logits) | {v['mean_offdiag_cosine_surface_logits']:.6f} |
| Mean off-diag cosine (alpha) | {v['mean_offdiag_cosine_alpha']:.6f} |
| Reconstruction succeeds (diagnostic) | {v['reconstruction_succeeds']} |
| Verdict | {v['label']} |

Full detail: `docs/FIELD_PROBE.md`, `logs/field_probe_report.json`.

## Pass / fail reading

- **Surface-only byte-LM:** inter-prompt surface cosine ≈ 1, field separation absent or unread.
- **Field carrying structure:** alpha/persistence differ across prompts; diagnostic recovery from field features beats a mean baseline; surface logits also move.
- **Transformer regime without the farm:** CE/metrics look healthy but this probe fails — scale Θ or more CE steps will not buy persistence.

## Next levers (named, not built here)

1. Real **MERGE** into durable field structures
2. **Dual clocks** (fast surface tick / slow field tick)
3. **Adaptive spans** (content-aware packet duration)

Run: `PYTHONPATH=. python tools/probe_field_persistence.py`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_scale_section(scale_path: Path, report: dict) -> None:
    v = report["verdict"]
    section = f"""

## Persistence test (field farm vs CE-only)

CE down + fluent-looking metrics **without** field reconstruction still means
**Transformer regime without the farm.** Capacity that matters for ATOM is
persistent structured matter in the toroidal field, not parameter count alone.

Latest probe (`tools/probe_field_persistence.py`):

- Mean RMS after prompt / after 20 gen: {v['mean_field_rms_after_prompt']:.6f} / {v['mean_field_rms_after_20_gen']:.6f}
- Off-diag cosine surface logits / alpha: {v['mean_offdiag_cosine_surface_logits']:.6f} / {v['mean_offdiag_cosine_alpha']:.6f}
- Verdict: {v['label']}

See `docs/PERSISTENCE_TEST.md` and `docs/FIELD_PROBE.md`.
"""
    existing = scale_path.read_text(encoding="utf-8") if scale_path.exists() else ""
    marker = "## Persistence test (field farm vs CE-only)"
    if marker in existing:
        head = existing.split(marker)[0].rstrip()
        scale_path.write_text(head + "\n" + section, encoding="utf-8")
    else:
        scale_path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only ATOM field persistence probe")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--last-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-role-prime", action="store_true")
    args = parser.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"[probe] loading {ckpt} (read-only, eval, no optimizer)")
    model, meta = load_model(ckpt, device=args.device)
    print(
        f"[probe] d_model={meta['config'].get('d_model')} "
        f"n_modes={meta['config'].get('n_modes')} "
        f"step={meta['training_meta'].get('step')}"
    )

    trials = []
    for prompt in DEFAULT_PROMPTS:
        print(f"[probe] prompt={prompt!r}")
        trial = run_prompt_trial(
            model,
            prompt,
            n_gen=args.n_gen,
            use_role_prime=not args.no_role_prime,
            seed=args.seed,
        )
        s0 = trial["stats_after_prompt"]
        s1 = trial["stats_after_gen"]
        print(
            f"        RMS {s0['field_rms']:.6f} -> {s1['field_rms']:.6f} "
            f"(norm {s0['field_norm']:.4f} -> {s1['field_norm']:.4f}) "
            f"atoms {s0['n_atoms']} -> {s1['n_atoms']}"
        )
        print(f"        gen_preview={trial['generated_preview']!r}")
        trials.append(trial)

    sensitivity = pairwise_sensitivity(trials)
    print(
        f"[probe] mean off-diag cosine logits={sensitivity['mean_offdiag_cosine_surface_logits']:.6f} "
        f"alpha={sensitivity['mean_offdiag_cosine_alpha']:.6f}"
    )
    recon = reconstruction_probe(model, trials, last_k=args.last_k)
    print(
        f"[probe] reconstruction_succeeds={recon['reconstruction_succeeds']} "
        f"field_differs={recon['field_differs_across_prompts']}"
    )

    # Local time America/Vancouver label
    try:
        from zoneinfo import ZoneInfo

        now_local = datetime.now(ZoneInfo("America/Vancouver"))
        ts_local = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:  # noqa: BLE001
        ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " PT"

    per_prompt_json = []
    for t in trials:
        per_prompt_json.append(
            {
                "prompt": t["prompt"],
                "primed_text": t["primed_text"],
                "n_prompt_packets": t["n_prompt_packets"],
                "packet_payloads": t["packet_payloads"],
                "stats_after_prompt": t["stats_after_prompt"],
                "stats_after_gen": t["stats_after_gen"],
                "generated_preview": t["generated_preview"],
                "generated_bytes": t["generated_bytes"],
                "surface_byte_logits_norm": t["surface_byte_logits_norm"],
            }
        )

    report = {
        "timestamp_local": ts_local,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "n_gen": args.n_gen,
        "prompts": DEFAULT_PROMPTS,
        "per_prompt": per_prompt_json,
        "prompt_sensitivity": sensitivity,
        "reconstruction": recon,
    }
    report["verdict"] = verdict_from(report)

    json_path = REPO_ROOT / "logs/field_probe_report.json"
    md_path = REPO_ROOT / "docs/FIELD_PROBE.md"
    persist_path = REPO_ROOT / "docs/PERSISTENCE_TEST.md"
    scale_path = REPO_ROOT / "docs/SCALE.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_jsonable(report), indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    write_persistence_test_doc(report, persist_path)
    append_scale_section(scale_path, report)

    print(f"[probe] wrote {json_path}")
    print(f"[probe] wrote {md_path}")
    print(f"[probe] wrote {persist_path}")
    print(f"[probe] updated {scale_path}")
    print(f"[probe] VERDICT: {report['verdict']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
