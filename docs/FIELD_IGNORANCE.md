# Field-ignorance loss

_Numbers only. English._

## Formula

```
L_ign = ReLU( cos( ℓ(α), ℓ(sg[α_bar]) ) − m )
```

- `ℓ(·)` = surface byte logits (same head as CE), from field `α` vs from stopgrad `α_bar`
- `α_bar` = sliding mean of the rolling α-bank of **other** prompts (not zero, not same-prompt)
- `m` = **0.85**
- weight = **0.08**

Purpose: penalize a surface that maps any non-zero α to one shared logit template.

## Flags

| Flag | Default | Disable |
|------|---------|---------|
| `--field-ignorance-weight` | `0.08` | `0` |
| `--field-ignorance-margin` | `0.85` | (n/a; weight 0) |

Wired in `AtomNativeModel._field_auxiliary_losses` next to field contrast. Reuses the existing α-bank from field-intel v2.

## Merge diagnostics (read-only)

When the merge path runs, log:

- `max_phase_coherence` — max pairwise phase-coherence **before** threshold filter
- `n_pairs_above_energy_floor` — pairs that clear the energy-floor checks

Thresholds `phase_coherence_threshold` / `merge_energy_floor` are **unchanged**.

## Smoke +2500 (2026-09-17)

See `docs/LIGN_SMOKE.md` for full numbers.

| metric | before | after |
|--------|--------|-------|
| logits cosine (inter-prompt) | 0.992294 | 0.991285 |
| α cosine | 0.720524 | 0.698694 |
| train `ign` (logged) | — | stuck **0.150** ⇒ cos_ign≈1.0 |
| merges | — | **0** |
| mph | — | **0.500** (threshold 0.55) |

**Fail** vs target logits ≪ 0.960. Next: inspect ℓ path (shared atom_r/persistence; bank closeness; hinge saturation) — one mechanism, no extra CE marathon.
