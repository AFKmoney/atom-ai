"""Run and document the atom-native ATOM experiment.

Usage:
    PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
      --data /tmp/wiki.train.raw --steps 500 \
      --output-dir checkpoints/atom_native_500

Streaming full-dataset ingest (no giant packet list in RAM):
    PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
      --stream --data-glob 'data/shards/*.txt' --steps 50000 \
      --output-dir checkpoints/atom_native_stream

The script intentionally uses the raw local corpus and never calls Hugging
Face or the legacy GPT-2 tokenizer.  It trains one transition per atom packet.
Episodes still pick random corpus starts in non-stream mode, but by default the
toroidal field is NOT wiped every short window (see ``--episode-length`` /
``--no-episode-reset``) so dialogue-scale persistence can accumulate.  Use
``--episode-reset`` to restore the old hard reset-every-episode behaviour.

In ``--stream`` mode the continuum is preferred: shards are read online and
``--no-episode-reset`` keeps the field alive across the infinite shard loop.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.atom_native import DEFAULT_ENERGY_DECAY_BOUNDS, AtomNativeModel
from src.io.atomizer import Atomizer
from src.io.stream_corpus import StreamingPacketSource, resolve_shard_paths

# Advisory flock so two run_atom_native.py processes do not hammer one box by accident.
# Default path (cwd-relative): checkpoints/atom-ai-train.lock
# Fallback if checkpoints/ cannot be created: /tmp/atom-ai-train.lock
DEFAULT_TRAIN_LOCK_PATH = Path("checkpoints") / "atom-ai-train.lock"
FALLBACK_TRAIN_LOCK_PATH = Path("/tmp/atom-ai-train.lock")


def acquire_train_lock(lock_path: Path | None = None):
    """Non-blocking exclusive flock; fail-fast if another train holds it.

    Returns an open file object that must stay open for the lock lifetime.
    Writes this process PID into the lock file for diagnostics.
    """
    candidates: list[Path] = []
    if lock_path is not None:
        candidates.append(Path(lock_path))
    else:
        candidates.append(DEFAULT_TRAIN_LOCK_PATH)
        candidates.append(FALLBACK_TRAIN_LOCK_PATH)

    last_error: Exception | None = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(path, "a+", encoding="utf-8")
        except OSError as exc:
            last_error = exc
            continue
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            holder = fh.read().strip() or "unknown"
            fh.close()
            raise SystemExit(
                f"error: another train holds the advisory lock\n"
                f"  lock path: {path.resolve()}\n"
                f"  holder:    {holder}\n"
                f"  this PID:  {os.getpid()}\n"
                f"Pass --allow-parallel-train to opt out (not recommended on one box)."
            ) from None
        except OSError as exc:
            fh.close()
            last_error = exc
            continue
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        os.fsync(fh.fileno())

        def _release(handle=fh, lock=path) -> None:
            try:
                if not handle.closed:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
            except (OSError, ValueError):
                pass

        atexit.register(_release)
        print(f"train lock acquired: {path.resolve()} (pid={os.getpid()})")
        return fh

    raise SystemExit(
        f"error: could not create/acquire train lock "
        f"(tried {[str(p) for p in candidates]}): {last_error}"
    )


PROMPTS = ["The future of AI is", "Bonjour, je m'appelle", "Valkyria Chronicles III is"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite_model(model: AtomNativeModel) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def transition_metrics(model: AtomNativeModel, current, target, train: bool) -> tuple[float, dict]:
    if train:
        loss, info = model.transition_loss(current, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite atom-native loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters), 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite atom-native gradient")
        return float(loss.detach().item()), {**info, "grad_norm": float(grad_norm)}
    with torch.no_grad():
        loss, info = model.transition_loss(current, target)
    return float(loss.item()), info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help="single corpus file (optional if --data-glob is set)",
    )
    parser.add_argument(
        "--data-glob",
        default=None,
        help="glob of shard files, e.g. 'data/shards/*.txt' (ordered lexicographically)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream shards online (required for large corpora / globs); "
             "do not encode the whole file into one packet list",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=1 << 20,
        help="UTF-8-safe read chunk size in stream mode (default 1048576)",
    )
    parser.add_argument(
        "--loop-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in stream mode, loop shards forever for an infinite continuum (default: loop)",
    )
    parser.add_argument(
        "--stream-buffer",
        type=int,
        default=2048,
        help="ring-buffer size of recent packets for stream-mode validation (default 2048)",
    )
    parser.add_argument(
        "--stream-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in --stream mode, prefetch next byte chunk on a background thread "
             "(default: on; use --no-stream-prefetch to disable)",
    )
    parser.add_argument(
        "--allow-parallel-train",
        action="store_true",
        default=False,
        help="opt out of the default advisory train lock "
             f"(default lock: {DEFAULT_TRAIN_LOCK_PATH})",
    )
    parser.add_argument(
        "--train-lock-path",
        default=None,
        help="override advisory lock path "
             f"(default: {DEFAULT_TRAIN_LOCK_PATH}; fallback {FALLBACK_TRAIN_LOCK_PATH})",
    )
    parser.add_argument("--output-dir", default="checkpoints/atom_native_chat")
    parser.add_argument("--checkpoint-name", default="atom_native.pt")
    parser.add_argument("--resume", default=None, help="optional checkpoint to resume")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=512,
        help="packets per random corpus window (default 512; was 64)",
    )
    parser.add_argument(
        "--episode-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, wipe toroidal/atomizer state at each episode boundary "
             "(default: keep field across windows for dialogue continuity)",
    )
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--n-atoms-max", type=int, default=128)
    parser.add_argument("--max-span-bytes", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--surface-learning-rate",
        type=float,
        default=None,
        help="optional higher LR for AtomSurfaceHead only (param group); "
             "default: same as --learning-rate",
    )
    parser.add_argument(
        "--field-max-rms",
        type=float,
        default=3.0,
        help="RMS saturate cap (default 3.0; prior chat runs used 4.0). "
             "Tighter cap reduces saturate wipe while keeping signal.",
    )
    parser.add_argument(
        "--energy-decay-min",
        type=float,
        default=DEFAULT_ENERGY_DECAY_BOUNDS[0],
        help=f"always-on energy_decay floor (default {DEFAULT_ENERGY_DECAY_BOUNDS[0]})",
    )
    parser.add_argument(
        "--energy-decay-max",
        type=float,
        default=DEFAULT_ENERGY_DECAY_BOUNDS[1],
        help=f"always-on energy_decay ceiling (default {DEFAULT_ENERGY_DECAY_BOUNDS[1]})",
    )
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument(
        "--atom-flush-every",
        type=int,
        default=256,
        help="clear structural atom list every N steps (0=never); keeps field. "
             "Bounds aggregation cost inside long episodes (default 256).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="print/record trajectory every N steps",
    )
    parser.add_argument(
        "--enable-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MERGE phase-coherent energetic atoms into heavier structure (default: on)",
    )
    parser.add_argument(
        "--field-loss-weight",
        type=float,
        default=0.05,
        help="weight for persistence probe (reconstruct packet features from α); 0 disables",
    )
    parser.add_argument(
        "--field-contrast-weight",
        type=float,
        default=0.1,
        help="weight for α contrastive hinge on surface logits; 0 disables",
    )
    parser.add_argument(
        "--field-contrast-margin",
        type=float,
        default=0.55,
        help="max allowed cosine(true_α_logits, null/shuffled_α_logits) before hinge",
    )
    parser.add_argument(
        "--field-ignorance-weight",
        type=float,
        default=0.08,
        help="weight for field-ignorance hinge L_ign=ReLU(cos(l(a),l(sg[a_bar]))-m); 0 disables",
    )
    parser.add_argument(
        "--field-ignorance-margin",
        type=float,
        default=0.85,
        help="margin m for field-ignorance cosine hinge (default 0.85)",
    )
    parser.add_argument(
        "--field-ignorance-ablate-shared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="for L_ign ℓ(α_bar) only: pass atom_r/persistence as None (zeros); "
             "ℓ(α) keeps live shared state (default: off)",
    )
    parser.add_argument(
        "--field-ignorance-prompt-bank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="build α_bar from fixed distinct prompts (probe-style), not train ticks "
             "(default: off)",
    )
    parser.add_argument(
        "--field-obligatory-hard",
        action="store_true",
        help="hard obligatory: mix floor=1.0 + freeze non-alpha CE bypass (implies readout)",
    )
    parser.add_argument(
        "--field-obligatory-readout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="main CE path: α-only residual into surface logits with non-zero floor "
             "(cannot bypass α via persist/atom_r alone; default: off)",
    )
    parser.add_argument(
        "--slow-every",
        type=int,
        default=1,
        help="run abstraction/consolidation every N ticks if field RMS stable (1=always)",
    )
    parser.add_argument(
        "--slow-rms-rel-tol",
        type=float,
        default=0.15,
        help="relative RMS change threshold for dual-clock slow path",
    )
    args = parser.parse_args()
    if not args.allow_parallel_train:
        acquire_train_lock(
            Path(args.train_lock_path) if args.train_lock_path else None
        )
    else:
        print("train lock skipped (--allow-parallel-train)")
    if not args.data and not args.data_glob:
        raise SystemExit("error: provide --data and/or --data-glob")
    if args.data_glob and not args.stream:
        raise SystemExit("error: --data-glob requires --stream (refuse full in-RAM encode of a shard glob)")
    if (args.energy_decay_min is None) != (args.energy_decay_max is None):
        raise ValueError("provide both --energy-decay-min and --energy-decay-max")
    energy_decay_bounds = (
        (float(args.energy_decay_min), float(args.energy_decay_max))
        if args.energy_decay_min is not None
        else DEFAULT_ENERGY_DECAY_BOUNDS
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stream_mode = bool(args.stream)
    shard_paths = resolve_shard_paths(args.data, args.data_glob)
    train_packets = None
    validation_packets = None
    packets = None
    raw = b""
    text = ""
    stream_source: StreamingPacketSource | None = None
    atomizer = Atomizer(max_span_bytes=args.max_span_bytes)

    if stream_mode:
        stream_source = StreamingPacketSource(
            atomizer,
            shard_paths,
            chunk_bytes=args.chunk_bytes,
            loop=args.loop_shards,
            buffer_size=args.stream_buffer,
            reset_atomizer=True,
            prefetch=bool(args.stream_prefetch),
        )
        # Peek one transition to fail fast on tiny/empty shards.
        try:
            first_pair = stream_source.next_transition()
        except StopIteration as exc:
            raise ValueError("stream source produced no packet transitions") from exc
        # Re-create source so training sees a clean continuum from the start.
        stream_source = StreamingPacketSource(
            Atomizer(max_span_bytes=args.max_span_bytes),
            shard_paths,
            chunk_bytes=args.chunk_bytes,
            loop=args.loop_shards,
            buffer_size=args.stream_buffer,
            reset_atomizer=True,
            prefetch=bool(args.stream_prefetch),
        )
        del first_pair
        data_bytes = sum(p.stat().st_size for p in shard_paths)
        metadata_lengths_placeholder = True
    else:
        if len(shard_paths) != 1 or args.data_glob:
            raise SystemExit("non-stream mode supports a single --data file only; use --stream for globs")
        data_path = shard_paths[0]
        raw = data_path.read_bytes()
        text = raw.decode("utf-8")
        packets = atomizer.encode_bytes(raw)
        if len(packets) < 4:
            raise ValueError("the corpus must yield at least four atom packets")
        if b"".join(packet.payload for packet in packets) != raw:
            raise AssertionError("atomization is not lossless")
        train_packets = packets[:-64] if len(packets) > 96 else packets[:-4]
        validation_packets = packets[len(train_packets) :]
        if len(train_packets) < 3 or len(validation_packets) < 2:
            raise ValueError("not enough packets for train/validation split")
        data_bytes = len(raw)
        metadata_lengths_placeholder = False

    seed_all(args.seed)
    model = AtomNativeModel(
        d_model=args.d_model,
        n_modes=args.n_modes,
        n_atoms_max=args.n_atoms_max,
        max_payload_bytes=args.max_span_bytes,
        atomizer=Atomizer(max_span_bytes=args.max_span_bytes),
        field_max_rms=args.field_max_rms,
        energy_decay_bounds=energy_decay_bounds,
        enable_merge=bool(args.enable_merge),
        slow_every=int(args.slow_every),
        field_loss_weight=float(args.field_loss_weight),
        field_contrast_weight=float(args.field_contrast_weight),
        field_contrast_margin=float(args.field_contrast_margin),
        field_ignorance_weight=float(args.field_ignorance_weight),
        field_ignorance_margin=float(args.field_ignorance_margin),
        field_ignorance_ablate_shared=bool(args.field_ignorance_ablate_shared),
        field_ignorance_prompt_bank=bool(args.field_ignorance_prompt_bank),
        field_obligatory_readout=bool(args.field_obligatory_readout) or bool(args.field_obligatory_hard),
        field_obligatory_hard=bool(args.field_obligatory_hard),
        slow_rms_rel_tol=float(args.slow_rms_rel_tol),
    )
    training_state = None
    start_step = 0
    if args.resume:
        training_state = model.load(args.resume)
        start_step = int((training_state or {}).get("step", 0) or 0)
        migrated = (training_state or {}).get("legacy_surface_migrated", False)
        dyn_load = (training_state or {}).get("dynamics_on_load") or {}
        print(
            f"resumed weights from {args.resume} "
            f"(recorded step={start_step}, legacy_surface_migrated={migrated})"
        )
        if dyn_load:
            print(
                f"dynamics on load: energy_decay="
                f"{dyn_load.get('energy_decay_before')} -> {dyn_load.get('energy_decay')} "
                f"(repaired={dyn_load.get('energy_decay_repaired')}, "
                f"bounds={dyn_load.get('energy_decay_bounds')})"
            )
        # CLI wins over ckpt config for obligatory readout (smoke / migrate).
        model.field_obligatory_readout = bool(args.field_obligatory_readout)
        model.surface.field_obligatory_readout = bool(args.field_obligatory_readout)
        if getattr(args, "field_obligatory_hard", False):
            model.field_obligatory_hard = True
            model.field_obligatory_readout = True
            model.surface.set_obligatory_hard(True)
            print("field_obligatory_hard=ON (mix_floor=1.0, non-alpha bypass frozen)")
        if args.field_obligatory_readout:
            print(
                "field_obligatory_readout=ON "
                f"(migrated={bool((training_state or {}).get('field_obligatory_migrated'))})"
            )
    elif args.field_obligatory_readout:
        model.field_obligatory_readout = True
        model.surface.field_obligatory_readout = True
        print("field_obligatory_readout=ON (fresh)")
    surface_lr = (
        float(args.surface_learning_rate)
        if args.surface_learning_rate is not None
        else float(args.learning_rate)
    )
    surface_ids = {id(p) for p in model.surface.parameters() if p.requires_grad}
    surface_params = [p for p in model.trainable_parameters if id(p) in surface_ids]
    other_params = [p for p in model.trainable_parameters if id(p) not in surface_ids]
    param_groups = [
        {"params": other_params, "lr": float(args.learning_rate)},
        {"params": surface_params, "lr": surface_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    print(
        f"optimizer groups: field/other lr={args.learning_rate} "
        f"({len(other_params)} tensors), surface lr={surface_lr} "
        f"({len(surface_params)} tensors); "
        f"energy_decay_bounds={energy_decay_bounds}; field_max_rms={args.field_max_rms}"
    )
    if stream_mode:
        print(
            f"stream mode: {len(shard_paths)} shard(s), chunk_bytes={args.chunk_bytes}, "
            f"loop_shards={args.loop_shards}, stream_buffer={args.stream_buffer}, "
            f"stream_prefetch={bool(args.stream_prefetch)}"
        )
    if training_state and "optimizer" in training_state:
        if training_state.get("legacy_surface_migrated"):
            print("optimizer restore skipped: surface LayerNorm migration changed bias scale")
        elif args.surface_learning_rate is not None:
            print("optimizer restore skipped: surface LR param groups differ from checkpoint")
        else:
            try:
                optimizer.load_state_dict(training_state["optimizer"])
                print("restored optimizer state")
            except Exception as exc:
                print(f"optimizer restore skipped: {exc}")
    model.train()

    if metadata_lengths_placeholder:
        metadata = {
            "stream": True,
            "data_path": str(args.data) if args.data else None,
            "data_glob": args.data_glob,
            "shard_paths": [str(p) for p in shard_paths],
            "data_bytes": data_bytes,
            "data_characters": None,
            "atomizer_version": atomizer.VERSION,
            "feature_dim": atomizer.feature_dim,
            "packet_count": None,
            "train_packet_count": None,
            "validation_packet_count": None,
            "mean_packet_bytes": None,
            "max_packet_bytes": None,
            "min_packet_bytes": None,
            "bytes_seen": 0,
            "config": {
                "d_model": args.d_model,
                "n_modes": args.n_modes,
                "n_atoms_max": args.n_atoms_max,
                "max_span_bytes": args.max_span_bytes,
                "episode_length": args.episode_length,
                "episode_reset": bool(args.episode_reset),
                "atom_flush_every": args.atom_flush_every,
                "learning_rate": args.learning_rate,
                "surface_learning_rate": surface_lr,
                "steps": args.steps,
                "start_step": start_step,
                "seed": args.seed,
                "surface_alphabet": 256,
                "field_max_rms": args.field_max_rms,
                "energy_decay_bounds": energy_decay_bounds,
                "stream": True,
                "chunk_bytes": args.chunk_bytes,
                "loop_shards": bool(args.loop_shards),
                "stream_buffer": args.stream_buffer,
                "stream_prefetch": bool(args.stream_prefetch),
                "enable_merge": bool(args.enable_merge),
                "field_loss_weight": float(args.field_loss_weight),
                "field_contrast_weight": float(args.field_contrast_weight),
                "field_ignorance_weight": float(args.field_ignorance_weight),
                "field_ignorance_ablate_shared": bool(args.field_ignorance_ablate_shared),
                "field_ignorance_prompt_bank": bool(args.field_ignorance_prompt_bank),
                "field_obligatory_readout": bool(args.field_obligatory_readout) or bool(getattr(args, "field_obligatory_hard", False)),
                "field_obligatory_hard": bool(getattr(args, "field_obligatory_hard", False)),
                "slow_every": int(args.slow_every),
            },
        }
    else:
        lengths = [len(packet.payload) for packet in packets]
        metadata = {
            "stream": False,
            "data_path": str(args.data),
            "data_glob": None,
            "shard_paths": [str(p) for p in shard_paths],
            "data_bytes": data_bytes,
            "data_characters": len(text),
            "atomizer_version": atomizer.VERSION,
            "feature_dim": atomizer.feature_dim,
            "packet_count": len(packets),
            "train_packet_count": len(train_packets),
            "validation_packet_count": len(validation_packets),
            "mean_packet_bytes": sum(lengths) / len(lengths),
            "max_packet_bytes": max(lengths),
            "min_packet_bytes": min(lengths),
            "config": {
                "d_model": args.d_model,
                "n_modes": args.n_modes,
                "n_atoms_max": args.n_atoms_max,
                "max_span_bytes": args.max_span_bytes,
                "episode_length": args.episode_length,
                "episode_reset": bool(args.episode_reset),
                "atom_flush_every": args.atom_flush_every,
                "learning_rate": args.learning_rate,
                "surface_learning_rate": surface_lr,
                "steps": args.steps,
                "start_step": start_step,
                "seed": args.seed,
                "surface_alphabet": 256,
                "field_max_rms": args.field_max_rms,
                "energy_decay_bounds": energy_decay_bounds,
                "stream": False,
                "enable_merge": bool(args.enable_merge),
                "field_loss_weight": float(args.field_loss_weight),
                "field_contrast_weight": float(args.field_contrast_weight),
                "field_ignorance_weight": float(args.field_ignorance_weight),
                "field_ignorance_ablate_shared": bool(args.field_ignorance_ablate_shared),
                "field_ignorance_prompt_bank": bool(args.field_ignorance_prompt_bank),
                "field_obligatory_readout": bool(args.field_obligatory_readout) or bool(getattr(args, "field_obligatory_hard", False)),
                "field_obligatory_hard": bool(getattr(args, "field_obligatory_hard", False)),
                "slow_every": int(args.slow_every),
            },
        }
    (output_dir / "atomization_stats.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    trajectory: list[dict] = []
    all_losses: list[float] = []
    field_rms_before_values: list[float] = []
    field_rms_after_values: list[float] = []
    field_scales: list[float] = []
    field_norm_values: list[float] = []
    start_time = time.perf_counter()
    episode = -1
    episode_start = 0
    # Soft cap: if atoms approach n_atoms_max, reset once to avoid unbounded growth
    # when --no-episode-reset keeps the field alive across windows.
    atom_soft_cap = max(16, int(args.n_atoms_max * 0.9))
    for step in range(1, args.steps + 1):
        current_episode = (step - 1) // args.episode_length
        if stream_mode:
            # Prefer continuous stream. Episode windows only force a field wipe
            # when --episode-reset is set; otherwise the field persists.
            if args.episode_reset and current_episode != episode:
                episode = current_episode
                model.reset_state(reset_atomizer=True)
                model.train()
            elif step == 1:
                episode = current_episode
                model.reset_state(reset_atomizer=True)
                model.train()
            else:
                episode = current_episode
            try:
                # Prefer iterator protocol (avoids extra Python method frame).
                current_packet, target_packet = next(stream_source)
            except StopIteration as exc:
                raise RuntimeError(
                    f"stream exhausted at step {step}/{args.steps}; "
                    "enable --loop-shards for infinite continuum"
                ) from exc
        else:
            if current_episode != episode:
                episode = current_episode
                if args.episode_reset or step == 1:
                    model.reset_state(reset_atomizer=True)
                else:
                    # Continuity-preserving compromise: at each window boundary, clear
                    # the structural atom list (keeps step cost bounded) but KEEP the
                    # toroidal field alpha + consolidation/abstraction persistence so
                    # dialogue-scale memory is not wiped every 64/512 ticks.
                    model.core.atoms = type(model.core.atoms)()
                    model.core.energy_history = []
                    if len(model.core.atoms) >= atom_soft_cap:
                        pass  # already cleared
                model.train()
                max_start = max(0, len(train_packets) - args.episode_length - 2)
                episode_start = random.randint(0, max_start) if max_start > 0 else 0
            index = episode_start + ((step - 1) % args.episode_length)
            if index >= len(train_packets) - 1:
                index = index % (len(train_packets) - 1)
            current_packet = train_packets[index]
            target_packet = train_packets[index + 1]

        optimizer.zero_grad(set_to_none=True)
        loss, info = transition_metrics(model, current_packet, target_packet, train=True)
        optimizer.step()
        if args.atom_flush_every and step % args.atom_flush_every == 0:
            model.core.atoms = type(model.core.atoms)()
            model.core.energy_history = []
        dynamics_state = model.stabilize_dynamics_parameters()
        model.core.training_loss = loss
        all_losses.append(loss)
        field_rms_before_values.append(info["field_rms_before"])
        field_rms_after_values.append(info["field_rms_after"])
        field_scales.append(info["field_scale"])
        field_norm_values.append(info["field_norm"])
        global_step = start_step + step
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": global_step,
                "run_step": step,
                "episode": episode,
                "loss": loss,
                "total_perplexity": float(np.exp(min(loss, 30.0))),
                "byte_loss": info["byte_loss"],
                "byte_perplexity": float(np.exp(min(info["byte_loss"], 30.0))),
                "length_loss": info["length_loss"],
                "field_norm": info["field_norm"],
                "field_rms_before": info["field_rms_before"],
                "field_rms_after": info["field_rms_after"],
                "field_scale": info["field_scale"],
                "grad_norm": info.get("grad_norm"),
                "n_atoms": info["n_atoms"],
                "merge_count": info.get("merge_count", 0),
                "merge_count_total": info.get("merge_count_total", 0),
                "slow_tick": info.get("slow_tick", True),
                "field_contrast_loss": info.get("field_contrast_loss", 0.0),
                "field_persist_loss": info.get("field_persist_loss", 0.0),
                "field_ignorance_loss": info.get("field_ignorance_loss", 0.0),
                "max_phase_coherence": info.get("max_phase_coherence", 0.0),
                "n_pairs_above_energy_floor": info.get("n_pairs_above_energy_floor", 0),
                "energy_decay": dynamics_state["energy_decay"],
                "coupling_scale": dynamics_state["coupling_scale"],
                "phase_sync": dynamics_state["phase_sync"],
            }
            if stream_mode and stream_source is not None:
                record["bytes_seen"] = stream_source.bytes_seen
                record["packets_seen"] = stream_source.packets_seen
                record["shard_index"] = stream_source.shard_index
                record["stream_epoch"] = stream_source.epoch
            trajectory.append(record)
            extra = ""
            if stream_mode and stream_source is not None:
                extra = (
                    f" bytes={stream_source.bytes_seen} "
                    f"pkts={stream_source.packets_seen} "
                    f"shard={stream_source.shard_index}"
                )
            print(
                f"step={global_step:07d} (run {step}/{args.steps}) "
                f"loss={loss:.5f} byte_ppl={record['byte_perplexity']:.3f} "
                f"atoms={record['n_atoms']} merges={record['merge_count_total']} "
                f"field={record['field_norm']:.4f} "
                f"rms={record['field_rms_after']:.4f} scale={record['field_scale']:.4f} "
                f"fcos={info.get('field_logit_cos_zero', float('nan')):.3f} "
                f"ign={info.get('field_ignorance_loss', 0.0):.4f} "
                f"mph={info.get('max_phase_coherence', 0.0):.3f}"
                f"{extra}"
            )
        if not finite_model(model):
            raise FloatingPointError(f"non-finite parameter at step {step}")

    elapsed = time.perf_counter() - start_time
    if stream_mode and stream_source is not None:
        metadata["bytes_seen"] = stream_source.bytes_seen
        metadata["packets_seen"] = stream_source.packets_seen
        metadata["stream_epoch"] = stream_source.epoch
        metadata["validation_packet_count"] = len(stream_source.validation_packets())
        (output_dir / "atomization_stats.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    checkpoint_path = output_dir / args.checkpoint_name
    model.save(
        checkpoint_path,
        extra_state={
            "step": start_step + args.steps,
            "run_steps": args.steps,
            "optimizer": optimizer.state_dict(),
            "trajectory": trajectory,
            "seed": args.seed,
            "episode_length": args.episode_length,
            "episode_reset": bool(args.episode_reset),
            "atom_flush_every": args.atom_flush_every,
            "stream": stream_mode,
            "bytes_seen": (stream_source.bytes_seen if stream_source else data_bytes),
        },
    )

    # Validation: non-stream uses held-out tail; stream uses ring-buffer samples.
    model.reset_state(reset_atomizer=True)
    model.eval()
    validation_records = []
    if stream_mode and stream_source is not None:
        val_pairs = stream_source.validation_transitions()
        # Cap validation work on huge rings.
        if len(val_pairs) > 256:
            val_pairs = val_pairs[-256:]
    else:
        val_pairs = list(zip(validation_packets[:-1], validation_packets[1:]))
    with torch.no_grad():
        for current, target in val_pairs:
            loss, info = transition_metrics(model, current, target, train=False)
            validation_records.append({"loss": loss, **info})
    if validation_records:
        validation_loss = float(np.mean([item["loss"] for item in validation_records]))
        validation_byte_loss = float(np.mean([item["byte_loss"] for item in validation_records]))
    else:
        validation_loss = float("nan")
        validation_byte_loss = float("nan")

    # Reload the final checkpoint before generation, then compare deterministic
    # samples so a restart cannot silently change the atom-native state path.
    reloaded = AtomNativeModel(
        d_model=args.d_model,
        n_modes=args.n_modes,
        n_atoms_max=args.n_atoms_max,
        max_payload_bytes=args.max_span_bytes,
        atomizer=Atomizer(max_span_bytes=args.max_span_bytes),
        field_max_rms=args.field_max_rms,
        energy_decay_bounds=energy_decay_bounds,
        enable_merge=bool(args.enable_merge),
        slow_every=int(args.slow_every),
        field_loss_weight=float(args.field_loss_weight),
        field_contrast_weight=float(args.field_contrast_weight),
        field_contrast_margin=float(args.field_contrast_margin),
        field_ignorance_weight=float(args.field_ignorance_weight),
        field_ignorance_margin=float(args.field_ignorance_margin),
        field_ignorance_ablate_shared=bool(args.field_ignorance_ablate_shared),
        field_ignorance_prompt_bank=bool(args.field_ignorance_prompt_bank),
        field_obligatory_readout=bool(args.field_obligatory_readout) or bool(args.field_obligatory_hard),
        field_obligatory_hard=bool(args.field_obligatory_hard),
        slow_rms_rel_tol=float(args.slow_rms_rel_tol),
    )
    reloaded.load(checkpoint_path)
    reload_state = {
        "n_atoms": len(reloaded.core.atoms),
        "field_norm": float(reloaded.core.state.alpha.detach().norm().item()),
        "time": float(reloaded.core.state.t.item()),
        "finite": bool(torch.isfinite(reloaded.core.state.alpha).all().item()),
    }
    generation = {}
    for prompt in PROMPTS:
        torch.manual_seed(args.seed + 1)
        source_bytes = model.generate_packets(
            prompt,
            max_packets=8,
            max_length=20,
            temperature=0.8,
            top_k=5,
            deterministic=False,
        )
        torch.manual_seed(args.seed + 1)
        reload_bytes = reloaded.generate_packets(
            prompt,
            max_packets=8,
            max_length=20,
            temperature=0.8,
            top_k=5,
            deterministic=False,
        )
        generation[prompt] = {
            "temperature": 0.8,
            "top_k": 5,
            "max_length": 20,
            "bytes_hex": source_bytes.hex(),
            "text": source_bytes.decode("utf-8", errors="replace"),
            "reload_text": reload_bytes.decode("utf-8", errors="replace"),
            "reload_match": source_bytes == reload_bytes,
        }

    metrics = {
        **metadata,
        "train": {
            "steps": args.steps,
            "start_step": start_step,
            "total_steps": start_step + args.steps,
            "episode_length": args.episode_length,
            "episode_reset": bool(args.episode_reset),
            "atom_flush_every": args.atom_flush_every,
            "elapsed_seconds": elapsed,
            "transitions_per_second": args.steps / max(elapsed, 1e-9),
            "first_loss": all_losses[0],
            "final_loss": all_losses[-1],
            "final_total_perplexity": float(np.exp(min(all_losses[-1], 30.0))),
            "finite_losses": all(np.isfinite(all_losses)),
            "finite_parameters": finite_model(model),
            "nan_or_inf": not all(np.isfinite(all_losses)) or not finite_model(model),
            "field_rms_before_max": max(field_rms_before_values),
            "field_rms_after_max": max(field_rms_after_values),
            "field_scale_min": min(field_scales),
            "field_norm_max": max(field_norm_values),
            "field_limited_steps": sum(scale < 0.999999 for scale in field_scales),
            "field_limited_fraction": sum(scale < 0.999999 for scale in field_scales) / len(field_scales),
            "final_dynamics": dynamics_state,
            "final_atoms_before_validation": len(reloaded.core.atoms),
            "stream": stream_mode,
            "bytes_seen": (stream_source.bytes_seen if stream_source else data_bytes),
        },
        "validation": {
            "transitions": len(validation_records),
            "loss": validation_loss,
            "total_perplexity": float(np.exp(min(validation_loss, 30.0))) if np.isfinite(validation_loss) else None,
            "byte_loss": validation_byte_loss,
            "byte_perplexity": float(np.exp(min(validation_byte_loss, 30.0))) if np.isfinite(validation_byte_loss) else None,
            "source": "stream_ring" if stream_mode else "held_out_tail",
        },
        "checkpoint": str(checkpoint_path),
        "reload": reload_state,
        "generation": generation,
    }
    (output_dir / "loss_trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
