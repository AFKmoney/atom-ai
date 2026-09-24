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

Defaults: `S_min=0.05`, `S_max=1.0`, `ε=1e-8`. Floor avoids silencing last_atom
when α collapses; ceiling avoids amplifying a weak last_atom above α.

### Flags (default OFF-compatible)

```
--last-atom-scale 1.0              # fixed S (unchanged; ignored if adaptive)
--last-atom-scale-adaptive         # enable S_eff equalization
--no-last-atom-scale-adaptive      # explicit off (default)
--last-atom-scale-min 0.05
--last-atom-scale-max 1.0
```

Live `train_until_coherent` recipe unchanged (adaptive OFF → fixed S=1.0).

### Falsification

Short probe (~8k) from 3808k lascale tip with free-run-aux ON + adaptive ON:
if chats stay letter-soup / dual-Bon attractor and/or effective la/α leaves ~1
after train, adaptive lever does not buy fluency either.
