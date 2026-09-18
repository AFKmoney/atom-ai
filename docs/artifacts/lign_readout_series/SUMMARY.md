# L_ign readout series — SUMMARY

_Generated 2026-09-17 19:22 PDT. Same src @ step 2102500. Each arm +2500 steps. No fluency claim._

## Question

Does fixing **how** ℓ(α) vs ℓ(α_bar) is formed drop inter-prompt **logits** cosine
below ~0.991 without killing **α** cosine (~0.70)?

## Table

| run | logits before | logits after | α before | α after | ign | mph | merges |
|-----|---------------|--------------|----------|---------|-----|-----|--------|
| baseline smoke | 0.992294 | **0.991285** | 0.720524 | 0.698694 | 0.150 | 0.500 | 0 |
| ablate-shared | 0.992294 | **0.991780** | 0.720524 | 0.698509 | 0.150 | 0.500 | 0 |
| prompt-bank | 0.992294 | **0.991156** | 0.720524 | 0.698703 | 0.000 | 0.500 | 0 |
| both | 0.992294 | **0.992358** | 0.720524 | 0.698481 | 0.000 | 0.500 | 0 |

## Answer

**No.** None of the three mechanism arms produced a meaningful logits drop vs ~0.991.
α stayed ~0.70 in all cases.

Side finding: **prompt-bank** (and **both**) drive train `ign→0` — α_bar from distinct
prompts is far enough that the cosine hinge is inactive (no L_ign gradient). Ablate-only
still saturates at ign=0.150 (cos_ign≈1).

## Artifacts

- `docs/artifacts/lign_ablate_2k5/`
- `docs/artifacts/lign_prompt_bank_2k5/`
- `docs/artifacts/lign_both_2k5/`
- `docs/artifacts/lign_readout_series/series_log.txt`
- Docs: `LIGN_ABLATION.md`, `LIGN_PROMPT_BANK.md`, `LIGN_ABLATION_PLUS_BANK.md`
