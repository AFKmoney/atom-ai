# L_ign ablation — shared atom_r / persistence OFF on ℓ(α_bar)

_English. Numbers only. No fluency claim. Generated 2026-09-17 19:22 PDT._

## What changed

Flag **`--field-ignorance-ablate-shared`** (default off; **ON** for this run).

When computing ℓ(α_bar) for `L_ign`, the surface call uses `alpha=α_bar` but
**`atom_r=None` and `persistent_state=None`** (zeros inside `field_features`).
ℓ(α) keeps the live shared state from the train forward. Same surface head.
Train α-bank unchanged (adjacent ticks).

## Setup

| Item | Value |
|------|-------|
| resume | `checkpoints/atom_native_lign_smoke/atom_native_src.pt` @ **2102500** |
| out | `checkpoints/atom_native_lign_ablate/atom_native.pt` @ **2105000** |
| steps | **2500** |
| L_ign | w=**0.08**, m=**0.85**, ablate=**ON**, prompt-bank=**OFF** |
| also | field-loss 0.08, contrast 0.45 / margin 0.45, merge ON, slow_every 4 |
| elapsed | **44.8s** (~55.7 tps) |

Artifacts: `docs/artifacts/lign_ablate_2k5/`.

## Probe (inter-prompt off-diag mean cosine)

| | before | after |
|--|--------|-------|
| surface logits | **0.992294** | **0.991780** |
| α (field) | **0.720524** | **0.698509** |

## Train (selected)

See `docs/artifacts/lign_ablate_2k5/train_log_excerpt.txt`.

- **ign**: early ~0.146 then sticks at **0.150** ⇒ cos_ign≈1.0 (same hinge saturation as baseline).
- **mph**: **0.500**; **merges**: **0**.

## Verdict vs baseline logits ~0.991

**Fail** — logits after **0.991780** (Δ vs before ≈ -0.0005; vs baseline after 0.991285 ≈ flat).
α not killed (**0.698509**). Ablating shared non-α inputs on the bar call alone did **not** break the shared logit template.
