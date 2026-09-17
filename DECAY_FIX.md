# Energy decay clamp fix — 2026-09-17 (PT)

## Problem

After the 1.05M persist train (`checkpoints/atom_native_chat_persist/atom_native.pt`),
`energy_decay` sat at **~0.001**.  Dynamics use:

```
decay = (energy_decay - 1) * alpha
```

so `0.001` → near-total wipe each tick.  The previous “safety” floor when
`energy_decay_bounds` was unset was only `[1e-3, 0.999]`, which *allowed* that
value.  Persist run metrics recorded `energy_decay_bounds: null`.

## Fix (always-on)

| Item | Choice |
|------|--------|
| Default band | **`[0.3, 0.95]`** (`DEFAULT_ENERGY_DECAY_BOUNDS`) |
| When applied | Every optimizer step **and** on `load()` |
| CLI | `--energy-decay-min/max` default to that band |
| On load of 1.05M ckpt | `0.001 → 0.3` (repaired flag in training state) |

Prior docs (`ATOM_FIELD_STABILIZATION.md`) used `[0.90, 0.999]`.  That ceiling
is still valid physics-wise; we chose a **wider but still strong** floor so the
parameter can learn without collapsing to wipe-mode.  Override via CLI if
needed.

## Related knobs (this retrain)

- **`field_max_rms=3.0`** (was 4.0).  Persist run only saturated ~2% of steps at
  4.0 (`field_scale_min≈0.997`), so saturate was not destroying signal; a
  modestly tighter cap still leaves headroom while limiting RMS blow-up under
  healthier decay.
- **Surface LR group**: `--surface-learning-rate 5e-4` vs field/other
  `--learning-rate 2.5e-4` (clean AdamW param groups on `model.surface`).
- Keep `--no-episode-reset`, `--atom-flush-every 256`, dialogue corpus.

## Retrain

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data data/corpus_train_dialogue.txt \
  --output-dir checkpoints/atom_native_chat_persist_v2 \
  --resume checkpoints/atom_native_chat_persist/atom_native.pt \
  --steps 250000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 --episode-length 512 --no-episode-reset \
  --atom-flush-every 256 --field-max-rms 3.0 \
  --learning-rate 2.5e-4 --surface-learning-rate 5e-4 \
  --energy-decay-min 0.3 --energy-decay-max 0.95 \
  --log-every 100 --seed 20260917
```

Log: `logs/train_decay_clamp_*.log`


## Retrain results (250k → 1.30M total)

| Metric | Persist 1.05M | Decay-clamp v2 1.30M |
|--------|---------------|----------------------|
| `energy_decay` | **0.001** | **0.300** (floor; stayed in band all run) |
| `field_max_rms` | 4.0 | 3.0 |
| field limited fraction | ~2.1% | **~98.9%** (almost always at cap) |
| val byte_ppl | ~28.4 | ~51.9 (worse) |
| train final loss | ~5.46 | ~4.53 |
| throughput | ~85 tr/s | ~73 tr/s |

Checkpoint: `checkpoints/atom_native_chat_persist_v2/atom_native.pt`  
Log: `logs/train_decay_clamp_250k.log`  
Chat samples: `logs/chat_retest_decay_clamp.txt`

### Honest chat verdict

French prompts still produce **embryonic / noisy** UTF-8 (apostrophe/`?` spam under
deterministic decode; slightly more letter soup under T=0.7). **Not** assistant-quality.
Clamp + surface LR fixed the dynamics bug but did **not** unlock coherent dialogue
in this 250k slice. High saturate fraction at RMS=3.0 + decay pinned to 0.3 may be
over-damping / over-clipping relative to the 1.05M operating point.

### Suggested next knob (not done here)

- Raise floor toward **0.5–0.7** (or init repair target mid-band) so decay is not
  always at the wipe-adjacent floor.
- Retry `field_max_rms=4.0` once decay is healthy, or raise saturate only when
  `field_limited_fraction` stays low.
