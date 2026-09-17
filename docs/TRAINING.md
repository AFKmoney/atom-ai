# Training — ATOM / atom-ai

### Minimal train

```bash
PYTHONPATH=. python tools/run_atom_native.py \
  --data data/samples/corpus_train_dialogue_slice.txt \
  --output-dir checkpoints/local_run \
  --steps 5000 \
  --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 \
  --episode-length 512 \
  --no-episode-reset \
  --atom-flush-every 256 \
  --field-max-rms 3.0 \
  --learning-rate 2.5e-4 \
  --surface-learning-rate 5e-4 \
  --energy-decay-min 0.3 --energy-decay-max 0.95 \
  --log-every 100 --seed 20260917
```

### Resume forever

```bash
PYTHONPATH=. python tools/run_atom_native.py \
  --data /path/to/your/dialogue_corpus.txt \
  --output-dir checkpoints/local_run \
  --resume checkpoints/local_run/atom_native.pt \
  --steps 250000 \
  ...same model flags...
```

`training.step` in the checkpoint is **cumulative**. Each resume adds `--steps`
on top of `start_step`. You can loop this indefinitely on CPU or GPU.

### Important flags

| Flag | Default (recommended) | Meaning |
|------|----------------------|---------|
| `--episode-length` | `512` | Packets per episode window (was 64 — too short for dialogue) |
| `--no-episode-reset` / `--episode-reset` | **no reset** | Persist field + consolidation across windows |
| `--atom-flush-every` | `256` | Soft clear atom *list* near cap without wiping field |
| `--field-max-rms` | `3.0` (chat) / `4.0` (older) | Cap field RMS |
| `--energy-decay-min/max` | **`0.3` / `0.95`** | Always-on clamp every step + on load |
| `--surface-learning-rate` | optional higher than field LR | AdamW group on `model.surface` |
| `--learning-rate` | `~2.5e-4`–`3e-4` after gen-fix | Field / non-surface params |

### Why these bounds?

- Episode length 64 + hard reset → field never survived a dialogue turn (`GEN_DEBUG.md`).
- Unclamped `energy_decay` drifted to `~0.001` or even negative → near-wipe per tick (`DECAY_FIX.md`).
- Default clamp `[0.3, 0.95]` is always applied; override only if you know why.

### Data

Ship only tiny samples. For real runs, build a UTF-8 dialogue corpus
(`Utilisateur:` / `Assistant:` lines help). See `data/CORPUS_DIALOGUE_INFO.txt`
and `docs/REPAIR_AND_TRAIN.md` for how the repair-session corpora were assembled
(Gutenberg FR plays + synthetic chat + seed) — rebuild outside this repo.

### Metrics honesty

Byte / packet perplexity and train loss are **not** “chat quality”. Report
prompt diversity, printable UTF-8 rate, and qualitative samples. Do not claim
assistant fluency from CE alone.

### Streaming full dataset (complete corpora, low RAM)

Do **not** load the whole corpus into one packet list. Use `--stream` with a
shard glob (or a single `--data` file):

```bash
PYTHONPATH=. python tools/run_atom_native.py \
  --stream \
  --data-glob 'data/shards/*.txt' \
  --chunk-bytes 1048576 \
  --loop-shards \
  --stream-buffer 2048 \
  --output-dir checkpoints/local_stream \
  --steps 50000 \
  --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 \
  --episode-length 512 \
  --no-episode-reset \
  --atom-flush-every 256 \
  --field-max-rms 3.0 \
  --learning-rate 2e-4 \
  --surface-learning-rate 6e-4 \
  --energy-decay-min 0.45 --energy-decay-max 0.95 \
  --log-every 100
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--stream` | off | Online ingest; required for `--data-glob` |
| `--data-glob` | none | Ordered shards, e.g. `data/shards/*.txt` |
| `--chunk-bytes` | `1048576` | UTF-8-safe read size |
| `--loop-shards` | on | Infinite continuum over shards |
| `--stream-buffer` | `2048` | Ring for validation samples |

Metadata records `stream`, globs, and `bytes_seen`. Prefer continuous stream
with `--no-episode-reset`; `--episode-reset` still wipes every episode window.

Capacity thesis: see [`SCALE.md`](SCALE.md) — GPT-3 *capability* without
GPT-3 *parameter/GPU-farm* scale via persistent structured matter.

