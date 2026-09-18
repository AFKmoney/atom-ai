# Field-ignorance loss

_Numbers only. English._

## Formula

```
L_ign = ReLU( cos( ℓ(α), ℓ(sg[α_bar]) ) − m )
```

- `ℓ(·)` = surface byte logits (same head as CE), from field `α` vs from stopgrad `α_bar`
- `α_bar` = sliding mean of the rolling α-bank of **other** prompts (not zero, not same-prompt)
  — or, with `--field-ignorance-prompt-bank`, mean of a **fixed distinct-prompt** α bank
- `m` = **0.85**
- weight = **0.08**

Purpose: penalize a surface that maps any non-zero α to one shared logit template.

## Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--field-ignorance-weight` | `0.08` | `0` disables |
| `--field-ignorance-margin` | `0.85` | hinge margin |
| `--field-ignorance-ablate-shared` | off | ℓ(α_bar) gets `atom_r`/`persistence` = None (zeros); ℓ(α) live |
| `--field-ignorance-prompt-bank` | off | α_bar from probe-style fixed prompts, not train ticks |

Wired in `AtomNativeModel._field_auxiliary_losses` next to field contrast.

## Merge diagnostics (read-only)

When the merge path runs, log:

- `max_phase_coherence` — max pairwise phase-coherence **before** threshold filter
- `n_pairs_above_energy_floor` — pairs that clear the energy-floor checks

Thresholds `phase_coherence_threshold` / `merge_energy_floor` are **unchanged**.

## Comparison (smokes +2500 from same src @ 2102500)

| run | logits cos (before→after) | α cos (before→after) | ign (final) | mph | merges |
|-----|---------------------------|----------------------|-------------|-----|--------|
| baseline L_ign | 0.992294→**0.991285** | 0.720524→0.698694 | **0.150** | 0.500 | 0 |
| ablate-shared | 0.992294→**0.991780** | 0.720524→0.698509 | **0.150** | 0.500 | 0 |
| prompt-bank | 0.992294→**0.991156** | 0.720524→0.698703 | **0.000** | 0.500 | 0 |
| both | 0.992294→**0.992358** | 0.720524→0.698481 | **0.000** | 0.500 | 0 |

**Success bar:** logits drop vs ~0.991 without killing α (~0.70). **Not met** on any of the three mechanism smokes.

Detail: `docs/LIGN_SMOKE.md`, `docs/LIGN_ABLATION.md`, `docs/LIGN_PROMPT_BANK.md`, `docs/LIGN_ABLATION_PLUS_BANK.md`, `docs/artifacts/lign_readout_series/SUMMARY.md`.
