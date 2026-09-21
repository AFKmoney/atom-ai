# ATOM architecture

A tick-based toroidal field. **Not a Transformer.**
No softmax attention, no BPE, no KV-cache, no HF tokenizer on the live path.
**New race: infinite, growable, CPU-scalable, live-modifiable.**

## One sentence

A UTF-8 byte becomes an atom; atoms live in a field α that RK4 advances;
the surface reads **the last atom + α** and emits the next byte.
The same `.pt` can be grown (d16→d32→d64…) and taught forever — stream, no forgetting.

## Why this is a new race (v2 proof: 0→1080k ms1M, d32, no patches)

**Growable to infinity:** `tools/grow_checkpoint.py` Net2Net-preserving:
- doubled output rows tiled+noise (symmetry breakers)
- doubled input cols zero-padded (old outputs exact)
- embeddings tiled+noise
- JL interleaved + frozen grown (α-branch exact on tiled α)
Result: d16 120k (val 1.352 / teacher 63.2%) → d32 grown init → 8k later 56% teacher, 16k 61.3% = d16 best. No restart. You can chain d32→d64→d128 same lineage. Field and readout are co-adapted — exact function ≠ same trajectory because d32 saturates at ~3000 steps vs ~850 d16 — but training re-adapts in 1 chunk.

**Infinitely learnable, not frozen:**
- Stream forever: `--stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 --stream-skip-packets k*steps` → fresh 64kB chunks each run.
- No replay (quarantined — drove CE 2.99→0.008 by reciting ring), no MERGE, no payload-copy (both off at train AND chat).
- `episode-reset --episode-length 8000` (was 6000 until 48k, 8000 sweet spot since 56k, tested 12000 worse): ~37.5% transient / 62.5% saturated for d32, exposes readout to both regimes. Episodes are far longer than saturation transient, so field persists across dialogue scales — not a wipe.
- `atom-flush-every 64` mandatory everywhere (train + primer probes) or atoms grow unbounded → 20× slowdown (same for probing).
- Same checkpoint lineage 0→120k d16 → 0→496k s21 → 504k→1M ms1M (s21 500k + s7 500k =1M) → 1080k 2nd epoch: val 1.765→0.551, teacher 56%→75.2%, no catastrophic forgetting. Val spikes at shard boundaries (1.645@632k, 1.49@976k) then descends — knowledge intact.

**CPU-scalable, GPU-fast:**
- No attention = O(n) not O(n²). Surface = `α-MLP(α) + 0.3*frozen_JL(α) + last_atom(2-gram + dentate top-25% + φ)`.
- CPU: d16 50-72 tick/s, d32 37-50 tps. 8k steps = 3-4 min. 1M = 125 chunks = ~8h CPU. GPU would be 10-100× — 10s GB in hours.
- `field-contrast-weight 0` — dead hinge (grads zero, 3 forwards/step wasted), cut for free speedup. `efference-every 10` = 1 tick/10 predict own byte no-grad → commit → CE on gold next — ~0% tps cost, 800/8000 eff ticks verified, not guilty of collapse.

**Live-modifiable without breaking + live survey:**
- `.pt` is infinitely trainable: `PYTHONPATH=. .venv/bin/python tools/run_atom_native.py --resume your.pt --data your_corpus.txt ...` on CPU, even after GPU pre-train on 10s GB.
- Read-only probes never stop training: `tools/probe_field_persistence.py` (field_rms cold ~0.04 / primed 0.26@560k→0.43@768k→0.62@872k→0.87@1M→1.06@1080k towards 3.0), `tools/probe_field_regimes.py --primer-packets 3500` (cold vs primed 3500), `tools/diagnose_teacher.py --max-span-bytes 1` (must, default span-4 gives ~22% artefact). Probe shows field differs across prompts but surface logits nearly identical when field not reliably read — Transformer-regime risk. Our scaling improves it.
- You can watch the model live while it trains — `field_rms`, `n_atoms`, generation `Peut-être`, `Peux qu'`, `Parler,`, `Je il`, clean stop `\n\n`.

## One sentence (original)

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
