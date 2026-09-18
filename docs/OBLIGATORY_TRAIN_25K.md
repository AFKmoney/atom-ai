# OBLIGATORY_TRAIN_25K — mid-train CE with field-obligatory readout

_Updated: 2026-09-17 ~20:15 PDT_

English. Numbers only. No fluency claim. No MERGE retune.

## Question

Does longer CE (**25 000** steps) with `--field-obligatory-readout` **preserve** the
+1500 smoke gain (logits **0.865**, α **0.705**), or does CE **re-collapse**
surface logits toward Part A (~0.98) / L_ign-series (~0.99)?

Success bar (from GO): logits **≤ 0.865**, ideally toward migrate-only **0.753**,
without killing α (~0.70).

## Setup

| Item | Value |
|------|-------|
| resume | stream `atom_native.pt` @ **2 900 000** → `checkpoints/atom_native_obligatory_train/atom_native_src.pt` |
| out | `checkpoints/atom_native_obligatory_train/atom_native.pt` @ **2 925 000** |
| steps | **25000** |
| data | `data/corpus_train_chat.txt` |
| shape | d=64, modes=64, n_atoms_max=512 |
| flag | `--field-obligatory-readout` |
| L_ign | **0** (isolated; no ablate/bank) |
| also | no-episode-reset, atom_flush=256, field_max_rms=3, energy_decay [0.45,0.95], field-loss 0.08, contrast 0.45, merge ON, slow_every 4, lr 1e-4 / surface 3e-4 |

Log: `logs/obligatory_train_25k.txt`. Artifacts: `docs/artifacts/obligatory_train_25k/`.

## Probe before → after

| Metric | Part A (stream 2.9M) | Migrate-only | Obl +1500 | **Obl +25k** |
|--------|----------------------|--------------|-----------|--------------|
| Surface logits off-diag cos | **0.979936** | **0.753479** | **0.865194** | **0.991599** |
| α off-diag cos | **0.706717** | 0.706717 | **0.705011** | **0.779656** |
| Mean RMS after prompt | 0.005340 | ~0.00534 | 0.005274 | 0.005149 |
| Mean RMS after 20 gen | 0.026840 | — | 0.024369 | 0.028981 |
| Verdict label | field carrying | — | field carrying | field carrying (partial) |

**Success bar:** FAIL — logits rose past +1500 and past Part A, into L_ign-series
territory (~0.992). α also drifted up (0.705 → 0.780).

## Train notes

- Elapsed ~537 s (~46.5 pkt/s). Finite losses/params.
- **merges=0** every logged step; **mph** mostly **0.500** (0.000 at step 1).
- Field RMS-limited most of the run (`field_limited_fraction` ≈ 0.957).
- Final train loss ~6.52; val byte_ppl ~51.7 (held-out tail).

## Chat samples (3 FR) — text only, no fluency claim

See `docs/artifacts/obligatory_train_25k/chat_samples.txt`.

- `Bonjour` → `Ei.-ebigrngh…`
- `Qui es-tu ?` → `"G<.kdoyyavo…`
- `Il était une fois` → `AT<-eblyygvh…`

## Verdict

**CE re-collapses the obligatory gain** under continued main-path training:
migrate-only 0.753 → +1500 0.865 → **+25k 0.992**. Frozen mix floor alone does
not stop long CE from driving surface logits toward a shared template. Next
lever (separate hypothesis, one at a time) must address CE vs floor interaction —
not stack L_ign.

## How to re-probe

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \
  --checkpoint checkpoints/atom_native_obligatory_train/atom_native.pt --skip-docs
```
