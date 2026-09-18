# L_ign both — ablate-shared + distinct prompt bank

_English. Numbers only. No fluency claim. Generated 2026-09-17 19:22 PDT._

## What changed

Both flags **ON**: `--field-ignorance-ablate-shared` and `--field-ignorance-prompt-bank`.

## Setup

| Item | Value |
|------|-------|
| resume | `atom_native_src.pt` @ **2102500** |
| out | `checkpoints/atom_native_lign_both/atom_native.pt` @ **2105000** |
| steps | **2500** |
| L_ign | w=**0.08**, m=**0.85**, ablate=**ON**, prompt-bank=**ON** |
| elapsed | **44.7s** (~55.9 tps) |

Artifacts: `docs/artifacts/lign_both_2k5/`.

## Probe

| | before | after |
|--|--------|-------|
| surface logits | **0.992294** | **0.992358** |
| α (field) | **0.720524** | **0.698481** |

## Train note

ign starts nonzero then falls to **0.000** (same hinge-off pattern as bank-only).
mph=**0.500**, merges=**0**.

## Verdict vs baseline logits ~0.991

**Fail** — after logits **0.992358** (no drop below ~0.991; slightly higher than before).
α ~**0.698481** (not killed). Combining both levers did not unlock
inter-prompt logit separation under this smoke budget.
