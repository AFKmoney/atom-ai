# OBLIGATORY_HARD — mix floor 1.0 + freeze non-α bypass

_Updated: 2026-09-18 (PT)_

English. Numbers only. No fluency claim. No MERGE retune. No L_ign. No stream marathon.

## Hypothesis (one mechanism)

Soft `--field-obligatory-readout` (mix floor **0.35**) **re-collapsed** under +25k CE
(logits **0.992**). CE can still route around the frozen α→logit branch via the
trainable **non-α** residual `(1−mix)·(decoder+skip)`.

**Hard mode** (`--field-obligatory-hard`):
1. Force obligatory ON.
2. Set **`obl_mix_floor = 1.0`** ⇒ mix weight ≡ 1 ⇒ surface logits = frozen α map.
3. **Zero + freeze** trainable non-α bypass params on the main CE surface path
   (`byte_decoder`, `length_decoder`, `field_to_state`, field skips, gates).
4. Keep frozen α→logit buffers (`alpha_byte_frozen` / `alpha_length_frozen` / JL).

Expected under short CE: logits cannot return to a shared template independent of α.
Unit: α vs `zeros_like(α)` (same `atom_r`) ⇒ logit cosine **≪ 0.99**.

## Status

Implementation + train/probe pending box PID capacity
(`/sys/fs/cgroup/pids.current` == `pids.max` == 19209 → `spawn bash EAGAIN`).

Soft +25k reference (already on main): logits **0.991599**, α **0.779656**.

## Measure plan (when shell recovers)

Resume stream `atom_native.pt` @ ~2.9M, hard ON, L_ign=0:
1. 5000 steps → probe logits/α
2. If not collapsed, +20000 (25k total) → probe again; else stop
3. Chat 3 FR prompts (text only)
