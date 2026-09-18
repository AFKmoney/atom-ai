# ATOM status snapshot — 2026-09-17

Repo: https://github.com/AFKmoney/atom-ai  
English docs only. No fluency claim.

## main (this push)

Science on tree:

- Field intel / contrast / MERGE MVP / dual clock
- Drive patch **A** (mode×content), HF `data.py` → `legacy/`
- `L_ign` (m=0.85, w=0.08) + merge mph logging — **falsified** as readout fix
- CPU perf: train lock default ON + stream one-ahead prefetch
- **L_ign readout levers:** `--field-ignorance-ablate-shared`, `--field-ignorance-prompt-bank` (falsified)
- **NEW: `--field-obligatory-readout`** — frozen α JL→logit mix on main CE path (`docs/READOUT_OBLIGATORY.md`)

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

## Next (scientific, one at a time)

1. Longer obligatory-aware train only if needed; keep frozen mix floor.
2. MERGE threshold only after mph evidence, separate commit.
3. Do not stack L_ign + obligatory without a new single hypothesis.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE · stacking multiple mechanisms · declaring fluent · force-push · retuning MERGE in the same change · restarting finished stream marathon.
