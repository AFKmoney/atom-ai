# Adaptive last_atom scale probe — 2026-09-24

English. Numbers. **No fluency claim.**

## Lever

```
S_eff = clamp(RMS(obl·α_MLP) / (RMS(la_raw)+ε), 0.05, 1.0)  # detached
logits += S_eff · la_raw
```

Commit feat: see git `feat: adaptive last_atom scale`.

## Recipe

- Base: `atom_native_step_3808000_d32_ms1M_lascale.pt`
- +8k → `atom_native_step_3816000_d32_ms1M_lascale_adapt.pt`
- free-run-aux every=10 H=4; `--last-atom-scale-adaptive`; ms1M d32 hard; allow-parallel
- Live `*_lrd.pt` / train_until_coherent **not** touched

## Table

| metric | before 3808k (fixed S=0.15) | after +8k adaptive |
|---|---|---|
| speech PASS | True (dual Bon-en) | True (::: / isi soup) |
| distinct | True | True |
| la/α mean (eff) | 3.51 (S=0.15) | 5.82 (S_eff=0.05 floor) |
| la raw RMS mean | ~58.9 | ~230 |
| mean S_eff | 0.15 fixed | 0.05 (always clamped) |
| attractor | Bon en / Assistant en | `:::` + `isisi` / Utilisiisir |

Adaptive preview at tip (no train): mean la/α≈1.35, S_eff≈0.052 — equalization works at freeze; after train last_atom grows through the floor.

## Honest read

- Broke prior Bon-en basin; replaced with worse colon/isi letter-soup.
- **Not fluent.** PASS is a weak latin/diversity gate only.
- Adaptive floor 0.05 insufficient once la_raw inflates under free-run-aux; effective imbalance worsened vs fixed S=0.15 tip.
- Lever does **not** buy fluency on this +8k probe.
