# ATOM status snapshot — 2026-09-17

Repo: https://github.com/AFKmoney/atom-ai  
English docs only. No fluency claim.

## main (post-doc push)

Science already on main before this note:

- Field intel / contrast / MERGE MVP / dual clock
- Drive patch **A** (mode×content), HF `data.py` → `legacy/`
- `L_ign` (m=0.85, w=0.08) + merge mph logging
- CPU perf: train lock default ON + stream one-ahead prefetch (`81d9444`)

## Measurements that matter

| Experiment | logits cos | α cos | notes |
|------------|------------|-------|-------|
| migrate-only (earlier) | ~0.938 | ~0.74 | readout partially alive |
| field_intel v1 | ~0.992 | ~0.32 | field separates; surface ignores |
| field_intel v2 +2500 | 0.938→0.960 | 0.738→0.579 | decoupling |
| **L_ign smoke +2500** | 0.992→0.991 | 0.721→0.699 | **L_ign did not move logits** |

## Open stream

Long `--stream` dialogue train may still run outside this repo checkout — leave it; soft-flush atoms, not field. rms=3 = limiter working.

## Next (scientific, one at a time)

1. Finish stream → ckpt + probe (no fluent judgment).
2. Fix **how** ℓ(α) vs ℓ(α_bar) is formed (see `LIGN_SMOKE.md`) — not more steps.
3. MERGE threshold only after mph evidence, separate commit.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE · stacking multiple mechanisms · declaring fluent.
