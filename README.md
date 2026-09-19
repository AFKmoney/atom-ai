# ATOM — Toroidal Fractal Intelligence

A tick-based field architecture. Not a Transformer.

```
bytes → Atomizer packets → toroidal atoms → field α (RK4) → surface bytes
```

## Status (2026-09-19)

**No fluency claim.** The field carries structure. The surface now emits
French *debris* (`Je`, `Jestiste`, `Ute`) on a d=16 byte-tick line.
It does not answer.

Measured log: [`docs/MEASURE_LOG.md`](docs/MEASURE_LOG.md).
Speech contract: [`docs/BYTE_TICK.md`](docs/BYTE_TICK.md).
Neuro bridges from `fractus-dataset`: [`docs/NEURO_BRIDGES_ATOM.md`](docs/NEURO_BRIDGES_ATOM.md).
Live snapshot: [`docs/STATUS.md`](docs/STATUS.md).

Rules: [`ATOM_RULES.md`](ATOM_RULES.md) — no HF tokenizer on the live path,
no attention softmax, no field wipe for CE.

## Speech contract (byte-tick)

Train and generate share the same graph: **1 octet = 1 atome = 1 tick**.

```
P(byte_t | bytes_<t, field, last atom, φ)
```

Hard readout:

```
logits = α-MLP(α) + 0.3·frozen_JL(α) + last_atom(last_byte, prev_byte, φ)
```

Payload-copy is **off**. `"generated"` boundary is **off**.
Generate uses n-gram anti-repeat so `Je Je Je` cannot run forever.

### Resume (sandbox tips, not in git)

| file | role |
|------|------|
| `atom_native_step_32000_dentate.pt` | best chat + anti-repeat |
| `atom_native_step_36000_big.pt` | best CE (1.61) |
| `atom_native_step_39500_hf.pt` | CATIE FR domain |

Do not resume `28000_replay` (CE 0.008 ring) or `36000_cont` (Je well).

```bash
PYTHONPATH=. python3 tools/run_atom_native.py \
  --stream --data data/corpus_fr_hf.txt --loop-shards \
  --resume checkpoints/byte_tick/atom_native_step_36000_big.pt \
  --steps 20000 \
  --d-model 16 --n-modes 16 --n-atoms-max 64 \
  --max-span-bytes 1 \
  --field-obligatory-hard --no-enable-merge
```

Chat probe:

```bash
PYTHONPATH=. python3 -c "
from src.atom_native import AtomNativeModel
m = AtomNativeModel(d_model=16, n_modes=16, n_atoms_max=64,
                    max_payload_bytes=1, field_max_rms=3.0,
                    field_obligatory_hard=True, enable_merge=False)
m.load('checkpoints/byte_tick/atom_native_step_32000_dentate.pt')
print(m.generate_packets('Qui es-tu ?', max_packets=48,
      deterministic=True, payload_copy=False,
      dialogue_wrap=True, speech_gate=False, merge_enabled=False))
"
```

## Vision

Not `neurons → layers → weight matrices → one-shot train`.

Token + context + state → toroidal structure. The structure interacts,
aggregates, abstracts, consolidates. Output is read from the living field.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md). Invariants in [`ATOM_RULES.md`](ATOM_RULES.md).

```
TOKEN STREAM → ATOMIZER → TOROIDAL ATOMS / FIELD α
        → RK4 → INTERACTION → AGGREGATION → ABSTRACTION
        → CONSOLIDATION → SURFACE BYTES
```

## Install / test

```bash
pip install -e .
python -m pytest test/ -q
```

## License

See repository license. Experimental research system.

## Citation

```bibtex
@misc{atom_toroidal_2026,
  title={ATOM: Toroidal Fractal Intelligence},
  author={PHIL},
  year={2026}
}
```
