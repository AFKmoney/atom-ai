# Adaptive last_atom S_min=0 probe — 2026-09-24

English. Numbers. **No fluency claim.**

## Lever

```
S_eff = clamp(RMS(obl·α_MLP) / (RMS(la_raw)+ε), 0.0, 1.0)  # detached; true RMS match
logits += S_eff · la_raw
```

Prior `S_min=0.05` falsified: under free-run-aux `la_raw` exploded so desired
S wanted below floor; effective la/α worsened. This probe drops the floor to 0.

Commit feat: `feat: adaptive last_atom S_min default 0 (true RMS match)`.

## Recipe

- Base: `atom_native_step_3816000_d32_ms1M_lascale_adapt.pt`
- +8k → `atom_native_step_3824000_d32_ms1M_lascale_adapt0.pt`
- free-run-aux every=10 H=4; `--last-atom-scale-adaptive` (new default S_min=0);
  ms1M d32 hard; allow-parallel
- Live `*_lrd.pt` / `train_until_coherent` **not** touched (tip advanced 4244k→4260k)

## Table

| metric | before 3816k (S_min=0.05) | preview S_min=0 @ freeze | after +8k S_min=0 |
|---|---|---|---|
| speech PASS | True (::: / isi) | — | True (::: / isi soup) |
| distinct | True | — | True |
| la/α mean (eff) | 5.82 (floor) | **1.00** | **1.00** |
| la raw RMS mean | ~230 | ~238 | ~458 |
| mean S_eff | 0.05 (clamped) | 0.0093 | 0.0106 |
| attractor | `:::` + isi / Utilisiisir | — | `:::` / colon-isi letter-soup |

## Honest read

- **Math succeeded:** S_eff tracked below former 0.05 floor; effective la contribution
  RMS matched α RMS exactly (mean ratio 1.00 before train preview and after +8k).
- `la_raw` still inflated under free-run-aux (~230→~458); scale compensated.
- Speech/chats remain letter-soup / colon-isi. **Not fluent.** PASS is a weak
  latin/diversity gate only.
- True RMS match does **not** buy French fluency on this +8k probe; imbalance was
  necessary but not sufficient. Next lever is elsewhere (not more S clamp tweaks).
