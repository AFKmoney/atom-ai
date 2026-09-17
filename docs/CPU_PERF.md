# CPU performance helpers (serialize + prefetch)

Lightweight box hygiene for ATOM training. **No science changes** — same
packets, same field, same loss. These knobs only reduce accidental CPU
contention and hide shard I/O latency.

## 1. Serialize trains (default ON)

Two `tools/run_atom_native.py` processes on one machine will thrash the same
CPU cores. By default the runner takes an **advisory exclusive flock** and
**fails fast** if another train already holds it (no infinite wait; does not
touch the other process).

| Item | Value |
|------|--------|
| Default lock path | `checkpoints/atom-ai-train.lock` (cwd-relative) |
| Fallback | `/tmp/atom-ai-train.lock` if `checkpoints/` cannot be created |
| Opt out | `--allow-parallel-train` |
| Override path | `--train-lock-path /path/to/lock` |

On conflict the process prints the lock path and holder PID (when available)
and exits non-zero. Lock is released on process exit.

## 2. Stream prefetch (default ON for `--stream`)

In `src/io/stream_corpus.py`, `StreamingPacketSource` can read the **next**
UTF-8-safe byte chunk on a background thread while the current chunk is
atomized / trained. One-ahead buffer only; same concatenation order and
bytes as without prefetch.

| Item | Value |
|------|--------|
| Default | ON when `--stream` |
| Disable | `--no-stream-prefetch` |
| Semantics | Identical packet payloads / order; no field reset |

## What NOT to do

These helpers are **not** a substitute for model or systems work you should
avoid on the ATOM path:

- **Do not** introduce DDP / multi-process data-parallel training as the
  “speed” fix — ATOM’s continuum is single-box structured matter, not a
  sharded parameter farm.
- **Do not** add attention stacks or other architecture rewrites under the
  banner of CPU perf.
- **Do not** wipe the toroidal field (or reset atomizer context) to “make
  steps cheaper” unless you explicitly want `--episode-reset` behaviour.
- **Do not** change L_ign / MERGE / dynamics / loss / thresholds here —
  science stays elsewhere.

Use serialize + prefetch first; profile before larger changes.
