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
