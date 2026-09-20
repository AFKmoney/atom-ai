# Released checkpoints

## Missing (not recoverable)

`3,425,000` (linguistic spans) and `3,458,000` (field_next) were trained into the same
`checkpoints/atom_native_obligatory_hard/atom_native.pt` path and **overwritten** by later
runs. Only metrics/docs remain under `docs/artifacts/`. Those weights are **gone**.

## Removed from RELEASE

Post-collapse tips (`3.498M`, `3.508M`, logits ≈0.999) — do not use for speech tests.

## Available now

| File | Step | Notes |
|------|------|-------|
| `atom_native_d32_32k_s21.pt` | 32,000 (s21) | **Active d32 tip**: val 1.490, teacher 63.0%, both regimes French |
| `atom_native_d16_120k_epr4.pt` | 120,000 | d16 tip (growth base): val 1.352, teacher 63.2% |
| `atom_native_2_925M_hard25k.pt` | 2,925,000 | Hard obligatory +25k (pre long hard-v2 cascade); probe logits were **0.861** |
| `atom_native_2_900M_stream.pt` | 2,900,000 | Stream dialogue tip before hard-v2 |

Not fluent — research snapshots only.
