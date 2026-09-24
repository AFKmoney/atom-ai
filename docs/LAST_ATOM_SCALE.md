# last_atom scale on hard mix — 2026-09-23

English. Numbers. **No fluency claim.**

## Problem

Hard mix: `logits = obl·α_MLP + 0.3·frozen_JL + payload + last_atom`.
After +32k free-run aux, last_atom/α logit RMS rose ~2.0→6.5; speech probe
collapsed to a shared `Peut-être…` attractor. last_atom dominates the mix.

## Math

```
logits_byte = obl * train_byte + 0.3 * frozen_byte + pay_byte + S * la_byte
```

`S = --last-atom-scale` (default **1.0** = current behavior). Choose
`S ≈ α_rms / la_rms` so the two trainable speech branches start comparable
(e.g. mean ratio 6.54 → S≈0.15). Gradients through α-MLP and last_atom both
remain; only the additive gain on last_atom changes.

## Falsification

Short probe (~4k–8k) from 3800k frroll tip with free-run-aux ON + S≈0.15:
if gen scraps / prompt diversity no better than S=1 tip, and/or attractor
returns while la/α (scaled) stays ~1, lever falsified for fluency.

## Flag (default OFF-compatible)

```
--last-atom-scale 1.0    # current behavior
--last-atom-scale 0.15   # example rebalance
```

Live `train_until_coherent` recipe unchanged (no flag → S=1.0).

## Adaptive scale (opt-in, 2026-09-24)

Fixed `S` was a one-shot guess (`α_rms/la_rms` at tip). After +8k with
`S=0.15`, la/α crept back to ~3.5× — fixed gain does not track drift.

### Math

```
α_term = obl * train_byte
la_raw = LastAtomReadout(last_b, prev_b, φ)
S_eff  = clamp( RMS(α_term) / (RMS(la_raw) + ε), S_min, S_max )
logits = obl·α_MLP + 0.3·frozen_JL + payload + S_eff · la_raw
```

RMS stats are **detached** (no grad through `S_eff` itself). Gradients still
flow through `α_MLP` and `LastAtomReadout` weights via the scaled add.

Defaults: `S_min=0.0`, `S_max=1.0`, `ε=1e-8`. Ceiling avoids amplifying a
weak last_atom above α. Floor is **0** so S_eff can true-RMS-match when
`la_rms ≫ α_rms` (see falsification of 0.05 below).

### Flags (default OFF-compatible)

```
--last-atom-scale 1.0              # fixed S (unchanged; ignored if adaptive)
--last-atom-scale-adaptive         # enable S_eff equalization
--no-last-atom-scale-adaptive      # explicit off (default)
--last-atom-scale-min 0.0          # true RMS match (was 0.05; falsified)
--last-atom-scale-max 1.0
```

Live `train_until_coherent` recipe unchanged (adaptive OFF → fixed S=1.0).

### Falsification of S_min=0.05 (2026-09-24)

+8k adaptive from 3808k lascale (`*_lascale_adapt.pt`): under free-run-aux,
`la_raw` exploded (~59→~230). Desired `S=α_rms/la_rms` wanted to go **below
0.05**; the floor kept too much last_atom → mean `S_eff` stuck at 0.05,
effective la/α worsened 3.51→5.82, speech worse (`:::` / isi soup). Frozen
preview at tip did equalize (~1.35) but only because tip already sat near the
floor. **Conclusion:** 0.05 floor falsified for free-run-aux train; next step
is true RMS match with `S_min=0` (keep `S_max=1.0`).

### Falsification (S_min=0)

Short probe (~8k) from 3816k lascale_adapt tip with free-run-aux ON + adaptive
ON + `S_min=0`: if chats stay letter-soup and/or effective la/α leaves ~1 after
train, even true RMS match does not buy fluency.
