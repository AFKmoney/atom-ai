# field_hard_couple S=1.0 +8k probe — 2026-09-24

English. Numbers. **No fluency claim.**

## Lever

`logits += S · W_hcouple · LN([mean‖std‖persist‖atom.r])` under hard.
`S=1.0` (4× prior S=0.25 probe). Dedicated `field_hard_couple_byte`.
No free-run-aux, no adaptive last_atom. Contrast stays 0. `field_loss=0.05`.

## Recipe

| item | value |
|------|-------|
| tip | `atom_native_step_4404000_d32_ms1M_lrd.pt` |
| out | `atom_native_step_4412000_d32_ms1M_lrd_ffield1.pt` |
| steps | +8000 |
| S | **1.0** |
| parallel | `--allow-parallel-train` vs live `*_lrd.pt` |
| elapsed | ~4528 s (~75 min wall under CPU contention w/ live) |

## Table

| ckpt | field_loss_w | couple S | PASS | distinct | chats (3) | α_rms | couple \|W\| mean |
|------|--------------|----------|------|----------|-----------|-------|-------------------|
| 4404k tip | 0.05 | 0 | False | False | Je/uuttilis ×3 | 0.0549 | 0.0160 (seeded, unused) |
| 4412k ffield1 | 0.05 | **1.0** | True* | True* | Oui,, / Je·uudiil / Je·uudiil | 0.0717 | **0.0172** |
| 4412k live peer | 0.05 | 0 | False | False | Je/uudirra ×4 probe | — | n/a |
| prior 4348k ffield S=0.25 | 0.05 | 0.25 | False | False | Peut-- ×3 | 0.0248 | 0.0161 |

\*PASS/distinct are **technical** (`probe_speech` rule: ≥2 prompts with ≥4 latin **and** strings differ). Bonjour→`Oui,,` (latin=3) while 3/4 stay `Je`/`uudiil*` — not fluent dialogue.

Attractors: `Je` + `uudiil*` / `Asssistant::::` (ffield1); live peer `Je`/`uudirra.`; older S=0.25 was `Peut--`. fcos=nan (contrast 0).

## vs S=0.25

| | S=0.25 @4340→4348k | S=1.0 @4404→4412k |
|--|--------------------|-------------------|
| tip speech | Peut-- locked | Je/uuttilis locked |
| after PASS | False | True* (thin) |
| \|W\| mean | 0.0161≈init | 0.0172 (+7.5% vs tip seed) |
| fluent? | no | **no** |

Stronger S moved W slightly more and broke identical-string lock via a Bonjour→Oui fragment, but surface remains attractor-dominated. CE still does not drive a real field→decode speech path in +8k.

## Honest read

Mechanism **live** (CLI banner S=1.0, cfg saved, `legacy_surface_migrated=False`, unit tests green, free_run_aux off).
+8k at S=1.0 did **not** unlock fluent / distinct chat vs tip intent.
`|W_hcouple|` still near init. **Not fluent.** Do not claim success.
Do **not** auto-stack free-run-aux / last-atom-adaptive on this evidence.
Falsified for fluency at S=1.0 / +8k on this tip (same family as S=0.25).

Live `train_until_coherent` / `*_lrd.pt` untouched by probe naming (`*_lrd_ffield1.pt`).
