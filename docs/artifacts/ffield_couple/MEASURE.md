# field_hard_couple +8k probe — 2026-09-24

English. Numbers. **No fluency claim.**

## Lever

`logits += S · W_hcouple · LN([mean‖std‖persist‖atom.r])` under hard.
`S=0.25`, dedicated `field_hard_couple_byte` (not frozen skip).
No free-run-aux, no adaptive last_atom. Contrast stays 0. `field_loss=0.05`.

## Recipe

| item | value |
|------|-------|
| tip | `atom_native_step_4340000_d32_ms1M_lrd.pt` |
| out | `atom_native_step_4348000_d32_ms1M_lrd_ffield.pt` |
| steps | +8000 |
| S | 0.25 |
| parallel | `--allow-parallel-train` vs live 5M |

## Table

| ckpt | field_loss_w | couple S | PASS | distinct | chats (3) | α_rms cold | couple \|W\| mean |
|------|--------------|----------|------|----------|-----------|------------|-------------------|
| 4340k tip | 0.05 | 0 | False | False | Peut-- / Peut-- / Je… | 0.0244 | n/a (no W) |
| 4348k ffield | 0.05 | 0.25 | False | False | Peut-- ×3 | 0.0248 | 0.0161 |
| 4348k live peer | 0.05 | 0 | False | False | Peut-- ×4 probe | — | n/a |

Attractors: `Peut--`, `aisssistant::::`. fcos=nan (contrast 0).

## Honest read

Mechanism **live** (CLI, unit tests, cfg saved, grads hit W in pytest).
+8k did **not** unlock distinct speech vs tip or vs live peer at same step.
`|W_hcouple|` stayed near init (~0.016) — CE did not move the new head much
under α-MLP + last_atom dominance. **Not fluent.** Do not claim success.
Falsified for fluency at S=0.25 / +8k; do not auto-stack without new hypothesis
(e.g. stronger S / couple-only CE aux / different feat slice).

Live `train_until_coherent` / `*_lrd.pt` untouched by this probe naming.
