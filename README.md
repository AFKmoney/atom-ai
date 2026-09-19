# ATOM — Toroidal Fractal Intelligence

A tick-based field. Not a Transformer.

```
byte → atom → field α (RK4) → next byte
```

**No fluency claim.** Debris (`Je`, `Jestiste`), not answers.

## Read these two

- **[TRAIN.md](TRAIN.md)** — how to train and probe
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the machine, in detail

Then: [docs/MEASURE_LOG.md](docs/MEASURE_LOG.md), [docs/STATUS.md](docs/STATUS.md),
[docs/BYTE_TICK.md](docs/BYTE_TICK.md).

## Install

```bash
pip install -e .
python -m pytest test/test_no_hf_tokenizer.py test/test_speech_lock.py -q
```

## One command

See `TRAIN.md`. Short form:

```bash
PYTHONPATH=. python3 tools/run_atom_native.py \
  --stream --data data/your.txt --loop-shards \
  --output-dir checkpoints/byte_tick \
  --steps 20000 --d-model 16 --n-modes 16 --n-atoms-max 64 \
  --max-span-bytes 1 --field-obligatory-hard --no-enable-merge
```

Copy every snapshot to a unique `atom_native_step_${STEP}.pt`.

## Banned

HF tokenizer · softmax attention · payload-copy · field wipe for CE ·
declaring fluent · mixing the 3.5M payload-copy weights into this graph.
