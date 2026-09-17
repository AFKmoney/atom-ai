# DRIVE_PROBE — token drive collinearity

Date: 2026-09-17 PT  
Probe: inject `"Bonjour"` vs `"xxxxxx"` from zero field; measure alpha RMS + cosine.
Model: AtomNativeModel d_model=32, n_modes=32, seed=0.

## Hypothesis (verified in `src/toroidal/dynamics.py` before patch)

| Claim | Verified |
|-------|----------|
| `mode_phase` regular polygon `arange * 2π/n_modes` | yes |
| token drive mean-broadcast identical on all modes | yes (pre-patch) |
| `rotated = sin(phase) * alpha` | yes |
| `_compute_phase_laplacian` exists but unused; linear Laplacian used | yes |

## Before (pre-patch numbers only)

```
RMS_Bonjour=0.00242920
RMS_xxxxxx=0.00243263
COSINE=0.97792614
```

## Patch choice: **A** (mode-modulated drive)

`drive_m,d = token_d * cos(mode_phase_m + 0.5 * token_d * d)` — not the same
vector on every mode. Same tensor shapes. No new module. Choice B (phase
Laplacian) not applied.

## After

```
RMS_Bonjour=0.00178140
RMS_xxxxxx=0.00177089
COSINE=0.95857894
```

Cosine dropped (0.9779 → 0.9586). RMS did not explode.

## energy_decay clamp

Train path calls `model.stabilize_dynamics_parameters()` after every
`optimizer.step()` (`tools/run_atom_native.py`); band default `[0.3, 0.95]`.
Already correct — no further clamp fix. See `DECAY_FIX.md`.

## Train path note

`tools/run_atom_native.py` supports `--stream`, `--resume`, `--atom-flush-every`
(soft atom list clear; field alpha not wiped), `--no-episode-reset` default.
No long train started for this probe.
