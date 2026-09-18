# ATOM status snapshot — 2026-09-17

Repo: https://github.com/AFKmoney/atom-ai  
English docs only. No fluency claim.

## main (this push)

Science on tree:

- Field intel / contrast / MERGE MVP / dual clock
- Drive patch **A** (mode×content), HF `data.py` → `legacy/`
- `L_ign` (m=0.85, w=0.08) + merge mph logging
- CPU perf: train lock default ON + stream one-ahead prefetch
- **L_ign readout levers:** `--field-ignorance-ablate-shared`, `--field-ignorance-prompt-bank`

## Measurements that matter

| Experiment | logits cos | α cos | ign | mph | merges | notes |
|------------|------------|-------|-----|-----|--------|-------|
| migrate-only (earlier) | ~0.938 | ~0.74 | — | — | — | readout partially alive |
| field_intel v1 | ~0.992 | ~0.32 | — | — | — | field separates; surface ignores |
| field_intel v2 +2500 | 0.938→0.960 | 0.738→0.579 | — | — | — | decoupling |
| **L_ign baseline +2500** | 0.992→**0.991** | 0.721→0.699 | 0.150 | 0.500 | 0 | hinge saturated |
| **ablate-shared +2500** | 0.992→**0.992** | 0.721→0.699 | 0.150 | 0.500 | 0 | no logits drop |
| **prompt-bank +2500** | 0.992→**0.991** | 0.721→0.699 | 0.000 | 0.500 | 0 | hinge off (too easy) |
| **both +2500** | 0.992→**0.992** | 0.721→0.698 | 0.000 | 0.500 | 0 | no logits drop |

Success metric (logits ≪ 0.991, α~0.70): **not achieved** on this series.

## Open stream

Long `--stream` dialogue train may still run outside this checkout — leave it; soft-flush atoms, not field. Do **not** retune MERGE here.

## Next (scientific, one at a time)

1. Finish stream → ckpt + probe (no fluent judgment).
2. If revisiting L_ign: hinge that stays active when bank negatives are strong (margin / divergence), **or** a different single readout coupling — not more CE steps alone.
3. MERGE threshold only after mph evidence, separate commit.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE · stacking multiple mechanisms · declaring fluent · force-push · retuning MERGE in the same change.
