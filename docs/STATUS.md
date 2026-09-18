# ATOM status snapshot — 2026-09-18

Repo: https://github.com/AFKmoney/atom-ai  
English docs only. No fluency claim.

## main (this push)

Science on tree:

- Field intel / contrast / MERGE MVP / dual clock
- Drive patch **A** (mode×content), HF `data.py` → `legacy/`
- `L_ign` (m=0.85, w=0.08) + merge mph logging — **falsified** as readout fix
- CPU perf: train lock default ON + stream one-ahead prefetch
- **L_ign readout levers:** `--field-ignorance-ablate-shared`, `--field-ignorance-prompt-bank` (falsified)
- **`--field-obligatory-readout`** — frozen α JL→logit mix on main CE path
  - smoke +1500: **PASS** (`docs/READOUT_OBLIGATORY.md`)
  - mid-train +25k: **FAIL / CE re-collapse** (`docs/OBLIGATORY_TRAIN_25K.md`)

## Stream 2.9M probe (finished marathon)

| Metric | Value |
|--------|-------|
| logits cos | **0.979936** |
| α cos | **0.706717** |
| RMS prompt / 20 gen | 0.005340 / 0.026840 |
| Verdict | field carrying structure (surface still under-reads) |

See `docs/STREAM_2_9M_PROBE.md`. Do **not** restart stream.

## Measurements that matter

| Experiment | logits cos | α cos | notes |
|------------|------------|-------|-------|
| migrate-only (earlier) | ~0.938 | ~0.74 | readout partially alive |
| field_intel v1 | ~0.992 | ~0.32 | field separates; surface ignores |
| **L_ign baseline +2500** | 0.992→**0.991** | 0.721→0.699 | hinge saturated — falsified |
| **ablate-shared +2500** | 0.992→**0.992** | 0.721→0.699 | no logits drop — falsified |
| **prompt-bank +2500** | 0.992→**0.991** | 0.721→0.699 | hinge off — falsified |
| **both +2500** | 0.992→**0.992** | 0.721→0.698 | no logits drop — falsified |
| **stream 2.9M (Part A)** | **0.980** | **0.707** | finished stream |
| **obligatory migrate-only** | **0.753** | 0.707 | flag ON, no train |
| **obligatory +1500** | 0.980→**0.865** | 0.707→**0.705** | **PASS** vs Part A |
| **obligatory +25k** | 0.865→**0.992** | 0.705→**0.780** | **FAIL** — CE re-collapse |

## Next (scientific, one at a time)

1. Hard-v2+ deeper α MLP +150k: probe OK (logits 0.914), chat still noise — deeper Linear→MLP alone did not unlock multi-word FR.
2. Next single hypothesis for phrase-level FR (do not stack MERGE retune / attention).
3. MERGE threshold only after mph evidence (this run: merges=0, mph≈0.5).
4. Do not stack L_ign + obligatory without a new single hypothesis.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE · stacking multiple mechanisms · declaring fluent · force-push · retuning MERGE in the same change · restarting finished stream marathon.

## Hard obligatory (2026-09-18)

| Experiment | logits | α | notes |
|------------|--------|---|-------|
| stream 2.9M | 0.980 | 0.707 | baseline |
| soft obligatory +25k | **0.992** | 0.780 | CE collapse — falsified |
| hard +5k | **0.776** | 0.763 | holds |
| hard +25k | **0.861** | 0.842 | **PASS** vs soft |

See `docs/OBLIGATORY_HARD.md`. Flag `--field-obligatory-hard`.


## Hard-v2 coherence run (2026-09-18)

| Metric | Value |
|--------|-------|
| start → end step | 2,925,000 → **3,125,000** (+200k) |
| final logits cos | **0.916** (≪0.99) |
| final α cos | **0.842** |
| chat | **not coherent FR** (short-token loops) |
| flag | `--field-obligatory-hard` (hard-v2 + RMS cap) |

See `docs/HARD_V2_COHERENCE.md`. Do not claim fluency.

## Hard-v2+ deeper α MLP (2026-09-18)

| Metric | Value |
|--------|-------|
| start → end step | 3,125,000 → **3,275,000** (+150k) |
| final logits cos | **0.914** (≪0.99) |
| final α cos | **0.840** |
| chat | **not coherent FR** (short-token noise) |
| mechanism | α-only MLP `JL → 4d → GELU → out` under hard |

See `docs/HARD_V2_DEEPER.md`. Do not claim fluency.
