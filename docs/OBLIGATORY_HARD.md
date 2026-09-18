# Field-obligatory HARD — 2026-09-18

English. Numbers. No fluency claim.

## Mechanism (one)

`--field-obligatory-hard` (implies readout):

1. `obl_mix_floor = 1.0` — CE cannot mix away from α
2. Non-α bypass residual **zeroed + frozen** (decoder / field_to_state / skips / gates)
3. Main CE path returns **frozen α JL→logit only** (`alpha_byte_frozen @ α_hat`)

Falsified earlier: soft obligatory (+25k logits→0.992), L_ign / ablate / bank.

## Runs (resume stream 2.9M)

| stage | steps from 2.9M | logits off-diag cos | α off-diag cos |
|-------|-----------------|---------------------|----------------|
| stream 2.9M (soft/off) | 0 | **0.980** | **0.707** |
| soft obligatory +25k | 25000 | **0.992** | 0.780 |
| **hard +5k** | 5000 | **0.776** | **0.763** |
| **hard +25k** (5k+20k) | 25000 | **0.861** | **0.842** |

## Verdict

**PASS vs soft collapse.** Hard keeps logits **≪ 0.99** after 25k CE (0.861). Mild rise 0.776→0.861 (frozen JL still adapts via field dynamics / α distribution), but CE cannot re-glue the shared surface template the way soft did.

Chat still noise — expected with pure frozen α map; not the success metric.

## Flags

```
--field-obligatory-hard
--field-ignorance-weight 0
```

Artifacts: `docs/artifacts/obligatory_hard_25k/`.
