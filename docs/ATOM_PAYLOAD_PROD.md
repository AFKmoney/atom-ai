# Atom-payload production under hard+MERGE — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Prior (required)

Field→next-packet @ `0ab4780`: hard+MERGE thr=0.45, nxp↓, chat still noise.
Do **not** reopen MERGE threshold / byte_decoder bypass without new evidence.

## Mechanism (one) — produce bytes from living atom payloads + α

Under `--field-obligatory-hard`, surface logits are no longer JL-fingerprint MLP
alone. Living atoms carry packet `payload` bytes (MERGE concatenates
constituents). Decode adds an **atom-native** branch:

| piece | detail |
|-------|--------|
| store | `ToroidalAtom.payload`; set on ingest; MERGE concatenates |
| pool | mean `byte_embed` over each atom payload; α-gate via `sigmoid(W a_hat)` |
| copy-bias | softplus scale toward bytes present (global hist + position-local) |
| gate | presence: exact α=0 ⇒ off (hard zeroing); nonzero α ⇒ full payload branch |
| place | **outside** α-MLP RMS cap (so copy is not crushed); still no `byte_decoder` reopen |
| keep | MERGE thr=0.45; `field_next_packet_weight=0.05`; frozen α map + capped α-MLP |

Unit test (`test/test_atom_payload_prod.py`): with distinct atom payloads, softmax
mass on those bytes ≫ empty-atom baseline; α=0 still ~0 logits; decoder frozen.

## Data / train

| item | value |
|------|-------|
| resume | hard ckpt @ **3,458,000** (after next-packet) |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | **+25k** (3.458M→**3.483M**) |
| hyperparams | hard ON, MERGE ON thr=0.45, nxp=0.05, linguistic@32, d=64, `field_max_rms=3` |
| throughput | ~30 tick/s CPU |

## Trajectory (3.458M → **3.483M** = +25k)

| stage | total step | nxp | merges (cum) | logits cos | α cos | chat |
|-------|------------|-----|--------------|------------|-------|------|
| next-pkt end | 3,458,000 | ~0.03 | 32870 | 0.838 | 0.676 | noise |
| payload +25k | **3,483,000** | **0.263→0.014** (last50 mean ~0.08) | **57772** | **0.999** | **0.645** | still noise |

nxp stays low. Merges keep firing. **Surface logits re-collapsed** (0.838→0.999)
while α still separates — payload copy from near-identical living atoms
(atom_flush → often 1 atom = last role-prime packet) homogenizes decode.

## Chat samples after +25k (honest — **not coherent**)

- **Bonjour** → `au  uatiur:i reroui e` scraps — no phrase
- **Qui es-tu ?** → near-empty — not an answer
- **Il était une fois** → `auiieatiui: …` scraps — not a story opener

**No fluency claim.**

## Probe

| metric | value |
|--------|-------|
| logits off-diag cos | **0.9986** |
| α off-diag cos | **0.645** |
| reconstruction | True (partial) |
| verdict | field carries structure; payload branch homogenized surface |

Artifacts: `docs/artifacts/atom_payload_prod/`.

## Verdict

**PASS** for mechanism: unit test green; payloads stored/merged; hard path uses
payload+α without reopening `byte_decoder`.

**FAIL** for speech: still noise after +25k; logits cos≈1.0 — **STOP**.

Do **not** auto-stack another +25k/150k. Next single hypothesis (one sentence):
**Retain a multi-atom payload ring (disable/raise `atom_flush_every`) so copy-bias
sees prompt-distinct merged bytes instead of one shared role-prime packet.**
