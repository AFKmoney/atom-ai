# Free-run auxiliary loss (L_roll) — 2026-09-23

English. Numbers. **No fluency claim.**

## Problem

Train optimizes teacher-forced next-byte CE on gold packets.
Chat runs `generate_packets`: commit model bytes, read next logits from the
free-run field. Proxy mismatch. 1-tick `--efference-every` is a partial bridge.

## Math (what is optimized)

On steps where `--free-run-aux-every N` fires (N>0):

```
L_roll = (1/H) Σ_{h=0}^{H-1} CE( logits(s_h), gold_h )
s_0    = current packet (living field as-is)
s_{h+1}= packet_from_payload( argmax( decode(logits(s_h)) ) )   # stop-grad commit
```

Same decode/commit graph as `generate_packets`. Gradients flow through each
step’s surface logits (α-MLP + frozen JL + last_atom under hard). Discrete
byte commits are non-differentiable (scheduled sampling). Weight W scales
`L_roll`; that step **replaces** ordinary CE (and supersedes 1-tick efference
when both would fire). Living field stays free-run-drifted after the step.

## What the gradient should do

Push surface params so that **under the model’s own commit trajectory**,
next-byte predictions stay close to the corpus continuation for H steps —
not only under teacher-forced gold history. Closes train↔chat mismatch.

## Falsification

After a short probe (2k–8k) from a tip copy with L_roll on:

1. Gen scraps on fixed prompts no more diverse / no less attractor-stuck than
   baseline tip → **falsified** for speech; try hard-mix last_atom rebalance next.
2. `free_run_horizon` logged as H and `free_run_aux_steps ≈ steps/N` → mechanism live.
3. Non-finite loss / grad → bug, not falsification of the idea.

## Flags (default OFF — live recipe unchanged)

```
--free-run-aux-every 0          # OFF
--free-run-aux-horizon 4
--free-run-aux-weight 1.0
```

Opt-in example: `--free-run-aux-every 10 --free-run-aux-horizon 4`.

## Code

`AtomNativeModel.free_run_aux_loss` in `src/atom_native.py`;
CLI + train loop in `tools/run_atom_native.py`; tests `test/test_free_run_aux.py`.
