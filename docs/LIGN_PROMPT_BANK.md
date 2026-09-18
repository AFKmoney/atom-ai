# L_ign distinct prompt bank — α_bar from fixed prompts

_English. Numbers only. No fluency claim. Generated 2026-09-17 19:22 PDT._

## What changed

Flag **`--field-ignorance-prompt-bank`** (default off; **ON** for this run).
Ablate flag **OFF**.

α_bar = mean of detached α snapshots from a fixed probe-style prompt set
(`DEFAULT_IGNORANCE_PROMPTS` in `src/atom_native.py`), refreshed every 256 ticks
with full train-state snapshot/restore. Train α-bank still updates for contrast;
L_ign no longer uses adjacent train ticks for α_bar.

## Setup

| Item | Value |
|------|-------|
| resume | `atom_native_src.pt` @ **2102500** (field_intel / lign_smoke src) |
| out | `checkpoints/atom_native_lign_prompt_bank/atom_native.pt` @ **2105000** |
| steps | **2500** |
| L_ign | w=**0.08**, m=**0.85**, ablate=**OFF**, prompt-bank=**ON** |
| also | same as prior L_ign smoke (contrast 0.45/0.45, merge ON, slow_every 4) |
| elapsed | **46.5s** (~53.8 tps) |

Artifacts: `docs/artifacts/lign_prompt_bank_2k5/`.

## Probe

| | before | after |
|--|--------|-------|
| surface logits | **0.992294** | **0.991156** |
| α (field) | **0.720524** | **0.698703** |

## Train note

**ign → 0.000** after warmup (hinge inactive). Distinct-prompt α_bar makes
`cos(ℓ(α), ℓ(α_bar))` drop **below** margin 0.85 immediately ⇒ **no L_ign gradient**.
mph=**0.500**, merges=**0**.

## Verdict vs baseline logits ~0.991

**Fail** on the success metric — after logits **0.991156** (still ~0.991).
α preserved (~**0.698703**). Mechanism side-effect: bank negatives
are “too easy” for the hinge (ign=0), so L_ign stops pushing.
