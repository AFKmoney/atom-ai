# Field → next-packet production loss — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Prior (required)

MERGE operational @ `d79d4a5`: thr 0.55→**0.45**, smoke +8k, **merges>0**.
Do **not** reopen MERGE threshold without new mph evidence.

## Mechanism (one) — predict next packet from α+atoms

Existing `field_probe` reconstructs the *current* packet features (persist).
Add a **separate** linear head `field_next_probe`:

| piece | detail |
|-------|--------|
| input | `field_feat_norm(field_features(α, persist, atom_r))` — α + atom_r |
| target | **next** (`target`) packet Atomizer features |
| loss | `1 − cos(pred, next_features)` |
| weight | `--field-next-packet-weight` **0.05** (small vs CE) |
| path | under `--field-obligatory-hard`; no decoder bypass / attention / HF |

Keep MERGE gates at operational defaults (thr=0.45, floor=0.08). Log `nxp=`.

Unit test: under hard, next-packet loss finite and grads hit `field_next_probe`.

## Data / train

| item | value |
|------|-------|
| resume | hard ckpt @ **3,433,000** (after MERGE smoke) |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | **+25k** (protocol 25k–50k; stop after first chunk) |
| hyperparams | hard ON, MERGE ON thr=0.45, `L_ign=0`, printable=0, nxp=**0.05**, linguistic@32, d=64, `field_max_rms=3` |
| throughput | ~38 tick/s CPU |

## Trajectory (3.433M → **3.458M** = +25k)

| stage | total step | nxp (1−cos) | merges (cum) | logits cos | α cos | chat |
|-------|------------|-------------|--------------|------------|-------|------|
| MERGE smoke end | 3,433,000 | (n/a) | 7968 | 0.860 | 0.705 | noise |
| next-pkt +25k | **3,458,000** | **0.947→0.031** (last50 mean ~0.10) | **32870** | **0.838** | **0.676** | still noise |

nxp learns (probe fits next features). Surface chat does **not** unlock phrases.

## Chat samples after +25k (honest — **not coherent**)

- **Bonjour** → scraps / loops — no phrase
- **Qui es-tu ?** → near-empty / scraps — not an answer
- **Il était une fois** → short scraps — not a story opener

**No fluency claim.**

## Probe

| metric | value |
|--------|-------|
| logits off-diag cos | **0.838** |
| α off-diag cos | **0.676** |
| reconstruction | True (partial) |
| verdict | field carrying structure; gen still noise |

Artifacts: `docs/artifacts/field_next_packet/`.

## Verdict

**PASS** for mechanism: next-packet aux trains (nxp↓) under hard + operational MERGE.
Speech still locked after +25k — do **not** claim fluency; do **not** auto-stack
another +25k without a new single hypothesis.
