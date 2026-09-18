# Printable aux on hard α-MLP — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Mechanism (one) — printable aux + hard-path chat bias

Hypothesis after deeper α-MLP +150k (logits **0.914**, chat short-token noise):
CE under frozen+MLP still places mass on binary/control bytes; decode-only
`prefer_printable` cannot teach the α-MLP. Add a **light train aux** on hard
logits that rewards printable UTF-8 / Latin-1-friendly mass, keep generation on
the hard α-MLP path, force `prefer_printable` in chat/generate.

Contract (unchanged hard-v2+):

1. Logits = `frozen_α_map(α) + cap * scale * α_MLP(α)` — **no** `byte_decoder` bypass
2. `printable_aux = (1 - mass) + 0.25 * (-log mass)` over surface byte softmax
3. Weight `--printable-aux-weight 0.08` (small vs CE)
4. Chat / `generate_packets`: re-assert `set_obligatory_hard(True)`; force
   `prefer_printable=True`; hard decode bias strength 2.5

Unit tests: aux grads hit α MLP; decoder stays frozen; hard decode flips mild NUL.

## Data / train

| item | value |
|------|-------|
| resume | `checkpoints/atom_native_obligatory_hard/` @ **3,275,000** (deeper MLP) |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | +50k planned ×2; **STOP after first +50k** (no multi-word FR) |
| hyperparams | hard ON, `L_ign=0`, printable_aux=0.08, `--no-episode-reset`, d=64, `field_max_rms=3` |
| throughput | ~36 tick/s CPU |

## Trajectory (3.275M → **3.325M** = +50k)

| stage | total step | logits cos | α cos | prn mass (train) | chat (honest) |
|-------|------------|------------|-------|------------------|---------------|
| deeper end | 3,275,000 | **0.914** | **0.840** | (n/a) | short-token noise |
| printable +50k | **3,325,000** | **0.912** | **0.837** | 0.872 → **0.890** | still scraps / loops |

Probe after +50k: logits **≪0.99** (0.912), α alive (0.837). No soft-collapse.
Printable mass rose only **+0.018** — aux moved the needle slightly; surface still
emits short-token noise.

## Chat samples after +50k (honest — **not coherent**)

- **Bonjour** → `'ns nps npu 'Diai.…` — letter scraps, no phrase
- **Qui es-tu ?** → multiline short tokens / loops — not an answer
- **Il était une fois** → `a… ecr… Oea…` — not a story opener
- **Bonjour (deterministic)** → `'ns 'ns '…s` loop

**STOP verdict:** after **+50k** printable-aux steps, generations do **not** show
multi-word readable French. Per protocol: do **not** burn another +50k/+150k on
the same lever. Field probe OK; printable mass slightly up; speech still locked.

No fluency claim.

## Next hypothesis (single)

Printable mass / short-byte CE is **not** the bottleneck for phrase-level FR.
Stronger next lever (pick one later):

1. **Teach Atomizer longer linguistic spans** (phrase/word boundaries aligned to
   dialogue `Assistant:` turns) so CE targets multi-byte readable spans instead of
   1–3 byte scraps.
2. **MERGE densification** only after mph evidence (this run: merges=0, mph≈0.5).

Do not stack attention / HF / MERGE retune / field wipe in the same change.

## Artifacts

`docs/artifacts/printable_aux/chunk_50k/` (train excerpt, probe, chat, run_metrics).
Train log: `logs/printable_aux_chunk_50k.txt`. Pytest: `docs/artifacts/printable_aux/pytest.txt`
(8 passed hard suite; full suite 46 passed / 1 skipped).

## Banned

Attention · HF tokenizer on live path · CE bypass via byte_decoder · stacking
MERGE retune · declaring fluent · force-push · `.pt` in git · burning another
blind CE chunk on printable aux alone.
