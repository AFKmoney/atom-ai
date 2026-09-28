# Branch b4828 — rollback to best-val 4828k, LR halved (2026-09-28)

## Why
- The long LR-decay run (`scripts/train_until_coherent.sh`, suffix `_lrd`) reached the
  5M target **without coherence** (`HIT_END_TARGET 5000000`).
- Best validation of the whole lrd run: **val 0.3944 @ 4828000**
  (`atom_native_step_4828000_d32_ms1M_lrd.pt`; next best 0.398 / 0.402).
- After 4828k the trajectory drifted back up: 4964k–4996k val ≈ 1.05–1.16, 5M val **0.898**,
  **0/3** coherence hits, chat collapsed onto the `Bonne` / `Oui,,,` attractor
  (4828k itself: `Je` / `Bonne` / `Bonne`, 0/3).
- The recipe is deterministic (per-chunk `--seed 20260913`, `--stream-skip-packets = START-496000`,
  optimizer state not restored because `--surface-learning-rate` is set), so resuming 4828k with
  the identical recipe would largely replay the old 4828k→5M drift.

## One lever
Same pattern that worked before (roll back to best-val, then decay LR): **halve both LRs**.

| | old `_lrd` | branch `_lrd_b4828` |
|---|---|---|
| `--learning-rate` | 3e-5 | **1.5e-5** |
| `--surface-learning-rate` | 7.2e-5 | **3.6e-5** |

Everything else identical: d32 / n-modes 32 / n-atoms-max 64, max-span-bytes 1,
field-obligatory-hard, no-enable-merge, no-payload-copy, last-atom-readout,
efference-every 10, atom-flush-every 64, episode-length 8000 + episode-reset, slow-every 1,
field-loss-weight 0.05, field-contrast-weight 0, allow-parallel-train, seed 20260913,
stream glob `data/corpus_fr_medium*.txt` with skip = START-496000, 8k-step chunks,
identical 3-prompt coherence probe. END_TARGET = 6,000,000.

## Implementation
`scripts/train_until_coherent.sh` is parametrized by env vars; defaults reproduce the original
`_lrd` run exactly (SUFFIX=lrd, LR=3e-5, SLR=7.2e-5, END_TARGET=5000000, no log tag):

- `SUFFIX` → ckpt `atom_native_step_<N>_d32_ms1M_${SUFFIX}.pt`, `run_metrics_<N>_d32_ms1M_${SUFFIX}.json`,
  train log `logs/train_d32_<A>_<B>_ms1M_${SUFFIX}.log`; tip auto-detect only globs this suffix.
- `START_CKPT` → `--resume` for the first chunk only (when no `${SUFFIX}` ckpt exists at START);
  START defaults to its step number.
- `LOG_TAG` (default `_${SUFFIX#lrd_}` = `_b4828`) → `logs/gen_coherent_<N>_b4828.txt`,
  `logs/coherence_watch_b4828.log`, `logs/COHERENT_STEP_b4828.txt` (old files untouched).

Old `_lrd` checkpoints (4836k…5000k) are never written or deleted by this branch.

## Launch / relaunch (idempotent: resumes from the newest `_lrd_b4828` tip)
```bash
cd /workspace/exports/atom-ai && SUFFIX=lrd_b4828 LR=1.5e-5 SLR=3.6e-5 END_TARGET=6000000 \
  START_CKPT=checkpoints/byte_tick/atom_native_step_4828000_d32_ms1M_lrd.pt \
  setsid nohup bash scripts/train_until_coherent.sh >> logs/continue_ms1M_b4828_master.log 2>&1 < /dev/null &
```
Master log: `logs/continue_ms1M_b4828_master.log`.
Exit codes: 0 = coherent (≥2/3 hits, step in `logs/COHERENT_STEP_b4828.txt`), 2 = hit 6M without coherence.

## Falsify
If val over the first few b4828 tips climbs back above ~0.6 or chat re-enters the `Bonne`
attractor with 0/3 hits, halving LR alone is not enough.
