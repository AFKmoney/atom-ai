# ATOM architecture

A tick-based toroidal field. **Not a Transformer.**
No softmax attention, no BPE, no KV-cache, no HF tokenizer on the live path.

## One sentence

A UTF-8 byte becomes an atom; atoms live in a field α that RK4 advances;
the surface reads **the last atom + α** and emits the next byte.

## Tick graph

```
UTF-8 stream
    |
    v
Atomizer          1 byte → AtomPacket {payload, features, boundary, phase}
    |
    v
Compiler          packet features → ToroidalAtom {r, φ, ω, E, payload}
    |
    v
Field core        α ∈ R^{n_modes × d}   RK4 + interaction + optional MERGE
    |
    v
Surface           logits[256] = α-MLP(α) + 0.3·JL(α) + last_atom(byte, prev, φ)
    |
    v
decode            argmax / sample one byte → commit as next atom
```

Train and generate **share this graph**. That is the speech contract
(`docs/BYTE_TICK.md`). Linguistic spans (5–32 bytes) broke it: train
saw a whole span, generate decoded slots independently.

## Pieces

### Atomizer (`src/io/atomizer.py`)

Cuts the byte stream into packets. Speech line: `max_span_bytes=1`.
Features are a fixed local observation (histogram, first/last nibble,
whitespace/alpha rates) — not a learned vocab.

`packet_from_payload` (generate commit) infers a train-like boundary
(`newline|punct|whitespace|max_span`). Tag `"generated"` is OOD and
must not be written.

### Toroidal atom (`src/toroidal/atom.py`)

One living unit:

| field | meaning |
|-------|---------|
| `r` | content vector |
| `φ` | phase (oscillatory binding) |
| `ω` | frequency |
| `E` | energy |
| `payload` | exact bytes this atom carries |

The collection is a list + stacked tensors. MERGE can fuse two atoms
when phase coherence is high. Speech line keeps MERGE **off**.

### Field α (`src/toroidal/state.py`, `dynamics.py`)

α is the persistent field, shape `(n_modes, d_model)`.
Dynamics integrate with RK4. Energy decay is bounded.
`field_max_rms` clips amplitude (sandbox: 3.0).

α is **not** a residual stream of a Transformer block. It is the
state that survives ticks, flushes, and (when you allow it) sessions.

### Interaction / aggregation / abstraction / consolidation

Still in the core. They are the structural story (atoms → aggregates
→ abstractions → persistent buffer). They are **not** the speech
readout. Speech is the surface.

### Surface / hard readout (`AtomSurfaceHead` in `src/atom_native.py`)

Under `--field-obligatory-hard`:

```
a_hat        = JL flatten(α)
train_logits = α-MLP(a_hat)          # trainable
frozen       = 0.3 · frozen_JL(a_hat) # buffer, not trained
last         = last_atom_readout(payloads, φ)
logits       = train + frozen + last
```

No RMS cap on the trainable branch (the cap re-collapsed CE).

**Last-atom readout** (corpus bridge #20 + #1):

1. embed last committed byte
2. embed previous byte (2-gram)
3. dentate: expand concat → ReLU → keep top 25% → project back
4. add `cos φ, sin φ` of the last atom
5. linear to 256 logits

This is **not** payload-copy (global hist / winner copy). Copy is off.

### Generate (`generate_packets`)

1. wrap prompt as `Utilisateur: …\nAssistant: ` (`src/speech_lock.py`)
2. ingest wrap packets into the live field
3. loop: predict one byte, commit `packet_from_payload`, append
4. n-gram anti-repeat: if the last 2–6 bytes already cycled, forbid
   that byte and draw once more
5. stop on empty, hard cycle, or `max_packets`

`speech_ok` is a phrase gate, not a language model. Leave
`speech_gate=False` while measuring.

### Train (`tools/run_atom_native.py`)

For each step: take packets `(current, target)`, `transition_loss` =
CE(next byte) + optional tiny aux, clip grad 1.0, step.

Optional **efference** (bridge #9): every 10 steps, commit the model's
own predicted byte, then CE on the gold target from that state.

Optional **replay** exists and is **off**. It drove CE to 0.008 by
reciting a ring.

Stream mode (`--stream`) reads shards online. Use it above ~200 kB
when `max_span_bytes=1` (otherwise encode holds one packet per byte).

## What ATOM is not

| banned | why |
|--------|-----|
| softmax attention / Flash / GQA | different machine |
| HF / GPT-2 tokenizer | Atomizer is the tokenizer |
| payload-copy attractor | surface copies instead of predicting |
| `"generated"` boundary | train/generate mismatch |
| field wipe to drop CE | destroys the thesis |
| stacking 4 new levers in one run | unreadable science |

## Neuro map (own corpus, not papers)

`docs/NEURO_BRIDGES_ATOM.md`. Wired: φ multiplex, dentate 2-gram,
efference. Replay implemented and gated off.

## Files that matter

```
src/atom_native.py          field + surface + generate
src/io/atomizer.py          packets
src/io/stream_corpus.py     online shards
src/speech_lock.py          wrap + gate
src/toroidal/*              atoms, RK4, MERGE, persist
tools/run_atom_native.py    trainer
tools/chat_atom_native.py   interactive
TRAIN.md                    how to run
docs/MEASURE_LOG.md         numbers
docs/STATUS.md              what is live
```

Legacy GPT-era: `src/main.py`, `src/training/trainer.py`, `legacy/`.
Do not start those for speech.
