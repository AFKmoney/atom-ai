# Field→hard-decode couple — 2026-09-24

English. Numbers. **No fluency claim.**

## Problem

Under `field_obligatory_hard`, the surface early-returns:

```
logits = obl·α_MLP(a_hat) + 0.3·frozen_JL(a_hat) + payload + S_la·last_atom
```

`a_hat` is α-only (`mean‖std`). Full spectral feats
`LN([mean‖std‖persist‖atom.r])` feed `field_to_state` / `field_byte_skip`
**before** that return, then are discarded. Persist + atom.r never reach
hard decode. `field_loss` (persist probe) trains a separate `field_probe`
head — not the hard logit path. Contrast stays **0** on the live recipe
(cut after the 88k collapse). Speech needs field→decode coupling, not
CE-only / α-MLP-only.

## Math (one opt-in lever)

```
feats   = LN([mean(α) ‖ std(α) ‖ persist ‖ atom.r])
couple  = W_hcouple · feats     # dedicated Linear, NOT frozen under hard
logits += S · couple            # + length twin
```

`S = --field-hard-couple-scale` (default **0** = OFF, bit-identical hard).
Hard zeros+freezes `field_byte_skip`; `W_hcouple` (`field_hard_couple_byte`)
is a new trainable head (init N(0,0.02)). When S>0, CE trains W_hcouple so
persist‖atom.r condition surface bytes.

## Why not contrast / raise field_loss / last_atom scale

| candidate | why not this turn |
|-----------|-------------------|
| raise `field_loss_weight` | already 0.05; CLI wins after `b420bc2`; trains probe≠decode |
| re-enable contrast | deliberately cut (88k collapse / dead hinge on this line) |
| last_atom scale | **closed** — prior probes; stop S clamps |
| free-run-aux + adaptive | stack noise; prefer field lever alone for A/B |

## Flag (default OFF — live recipe unchanged)

```
--field-hard-couple-scale 0.0    # OFF
--field-hard-couple-scale 0.25   # probe example
```

CLI wins after resume (same pattern as field_loss / last_atom_scale).

## Falsification

+8k from clean `*_lrd.pt` tip with S>0 alone (no free-run-aux, no adaptive S):
if speech PASS/distinct / 3 chats no better than tip baseline and couple
RMS stays decorative → lever falsified for fluency; do not auto-stack.

## Load note

Tip checkpoints predating this head lack `field_hard_couple_*` keys. Load seeds only those weights (`N(0,0.02)`); it must **not** reinit `field_feat_norm` / `field_to_state` (see fix commit). Legacy migrate flag stays false.

## Code

`AtomSurfaceHead.forward` hard branch; CLI in `tools/run_atom_native.py`;
tests `test/test_field_hard_couple.py`.

## Probe log (S series)

| S | tip→out | PASS | \|W\| mean | note |
|---|---------|------|-----------|------|
| 0.25 | 4340→4348k `*_ffield` | False | ~0.016 | Peut--; near init |
| **1.0** | 4404→4412k `*_ffield1` | True* | ~0.017 | Je/Oui fragment; **not fluent** |

Details: `docs/artifacts/ffield_couple/` and `docs/artifacts/ffield_couple_s1/`.
\*technical probe rule only — do not read as fluency.
