# READOUT_OBLIGATORY — field-obligatory surface path (main CE)

_Updated: 2026-09-17 ~20:00 PDT_

English. Numbers only. No fluency claim. No MERGE retune.

## Problem

Stream ckpt @ 2.9M (`STREAM_2_9M_PROBE`): α separates (off-diag cos **0.707**) but
surface logits stay high (**0.980**). Prior L_ign levers (hinge / ablate-shared /
prompt-bank / both) were **falsified** — they did not force the **main** CE
forward to depend on α.

## Mechanism (one new path)

**`field_obligatory_readout`** on `AtomSurfaceHead.forward` (same call CE uses):

1. **α-only JL fingerprint** of full flattened α (not mean‖std alone — that
   collapsed inter-prompt cos ~0.93 while full α sits ~0.71).
2. **Frozen** random α→logit map (`alpha_byte_frozen` / `alpha_length_frozen`
   buffers) mixed into logits with hard weight
   `mix ∈ [obl_mix_floor, 1]` (`obl_mix_floor=0.35`). CE cannot drive this to 0
   and cannot retrain the frozen map to a shared template.
3. **Trainable** α adapter (`alpha_byte_proj`) added with floored
   `softplus(obl_gate)+obl_floor` residual.
4. Skip path also fed α-only (padded) when the flag is on.

Flag: `--field-obligatory-readout` / `--no-field-obligatory-readout` (default off).
L_ign left at **0** for this smoke (different mechanism).

Unit test: same `atom_r`/persistence, α vs `zeros_like(α)` ⇒ logit cosine **< 0.99**
(`test/test_generation_smoke.py::FieldObligatoryReadoutTests`).

## Setup

| Item | Value |
|------|-------|
| resume | copy of stream `atom_native.pt` @ **2 900 000** → `checkpoints/atom_native_obligatory/atom_native_src.pt` |
| out | `checkpoints/atom_native_obligatory/atom_native.pt` @ **2 901 500** |
| steps | **1500** |
| data | `data/corpus_train_chat.txt` |
| flag | `--field-obligatory-readout` |
| L_ign | **0** (isolated) |
| also | field-loss 0.08, contrast 0.45/0.45, merge ON, slow_every 4 |

Artifacts: `docs/artifacts/obligatory_smoke_1k5/`.

## Before / after probe

| Metric | Part A (stream 2.9M) | Migrate-only (obligatory ON) | After +1500 |
|--------|----------------------|------------------------------|-------------|
| Surface logits off-diag cos | **0.979936** | **0.753479** | **0.865194** |
| α off-diag cos | **0.706717** | 0.706717 | **0.705011** |
| Mean RMS after prompt | 0.005340 | ~0.00534 | 0.005274 |
| Mean RMS after 20 gen | 0.026840 | — | 0.024369 |
| Verdict | field carrying structure | — | field carrying structure |

**Success bar:** logits drop vs Part A without killing α → **PASS**
(0.980 → 0.865; α ~0.71 unchanged).

Migrate-only already separates (0.753); short CE raises logits somewhat
(0.865) but stays well below Part A and far below the L_ign series (~0.991).

## Not done

- No MERGE threshold retune
- No long CE / stream restart
- No fluency claim

## How to re-probe

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \
  --checkpoint checkpoints/atom_native_obligatory/atom_native.pt --skip-docs
```
