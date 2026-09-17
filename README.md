# ATOM / atom-ai

A research architecture for **atom-native continuous structured learning** on a
**persistent toroidal field** — not a Transformer, not GPT-2 embeddings, not
token-ID cross-entropy over a BPE vocabulary.

```text
UTF-8 bytes → Atomizer packets → ToroidalAtom → field / RK4 / interaction
             → aggregation → abstraction → consolidation → surface decode
```

Package name on PyPI/layout: **`atom-ai`**. Imports stay as `from src...`
(flat `src/` layout) to avoid breaking existing tools and checkpoints.

---

## Why not a Transformer?

Transformers flatten a window, apply token–token attention, and treat IDs as
the fundamental unit. ATOM treats each input tick as **computational matter**:

| Transformer world | ATOM world |
|-------------------|------------|
| subword ID → embedding table | raw UTF-8 span → reversible `AtomPacket` |
| MultiheadAttention / QKV | local toroidal interaction + RK4 dynamics |
| wipe state between batches | persistent field + consolidation (by default) |
| next-token CE on vocab | next-**packet** surface over bytes |

Hard rules: [`ATOM_RULES.md`](ATOM_RULES.md) · tokenization: [`ATOM_NATIVE_TOKENIZATION.md`](ATOM_NATIVE_TOKENIZATION.md).

---

## Quickstart

```bash
cd atom-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Smoke tests
PYTHONPATH=. pytest test/ -q

# Tiny train (CPU-friendly)
PYTHONPATH=. python tools/run_atom_native.py \
  --data data/samples/corpus_train_dialogue_slice.txt \
  --output-dir checkpoints/local_smoke \
  --steps 200 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --episode-length 512 --no-episode-reset \
  --field-max-rms 3.0 --log-every 50

# Chat with the shipped ~8 MB sample (not fluent — demo only)
PYTHONPATH=. python tools/chat_atom_native.py \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt \
  --prompt "Bonjour"

# Generation smoke
PYTHONPATH=. python tools/smoke_gen_check.py \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt
```

Resume forever (recommended flags): see [`docs/TRAINING.md`](docs/TRAINING.md).

---

## Layout

```text
atom-ai/
├── src/
│   ├── atom_native.py      # AtomCompiler + AtomNativeModel + surface head
│   ├── io/atomizer.py      # UTF-8 → AtomPacket (no HF)
│   ├── toroidal/           # field, RK4, interaction, aggregation, …
│   ├── training/           # legacy sequential trainer (toroidal core tests)
│   └── main.py             # CLI → train / chat / interactive
├── tools/
│   ├── run_atom_native.py
│   ├── chat_atom_native.py
│   └── smoke_gen_check.py
├── test/                   # atomizer, gen smoke, core invariants, resume
├── data/samples/           # tiny dialogue slices
├── checkpoints/            # one ~8 MB persist sample + README
└── docs/                   # architecture, training, chat, antislop, log
```

---

## Documentation map

| Doc | Content |
|-----|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Atomizer → AtomNative → toroidal pipeline |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Flags, decay bounds, resume forever |
| [`docs/CHAT.md`](docs/CHAT.md) | How to chat + honest limitations |
| [`docs/MANIPULATIONS.md`](docs/MANIPULATIONS.md) | Chronological Grok/repair session log |
| [`docs/ANTISLOP.md`](docs/ANTISLOP.md) | What was rejected and why |
| [`docs/GEN_DEBUG.md`](docs/GEN_DEBUG.md) | Surface bias, episode reset, boundary |
| [`docs/DECAY_FIX.md`](docs/DECAY_FIX.md) | `energy_decay` clamp `[0.3, 0.95]` |
| [`docs/REPAIR_AND_TRAIN.md`](docs/REPAIR_AND_TRAIN.md) | Quarantine + corpus + train notes |
| [`CHANGELOG.md`](CHANGELOG.md) | Dated entries |

All documentation is **English-only**.

---

## Status (honest)

Experimental. After ~1.05M CPU steps on dialogue data, generation is
**prompt-sensitive** and mostly printable UTF-8, but **not** fluent French/English
chat. Success metric ≠ GPT parity; success = atom-native spine that trains and
decodes without Transformer/GPT-2 crutches.

---

## License

Proprietary — research and evaluation.
