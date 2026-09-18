# Hard-v2 α-trainable head → coherence attempt — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters.

## Mechanism (one) — hard-v2

`--field-obligatory-hard` upgraded in place:

1. `obl_mix_floor = 1.0`; non-α decoder / skip / gates **zeroed + frozen**
2. Forward:
   `logits = frozen_α_map(α) + cap * obligatory_scale() * α_*_proj(α)`
   with `cap = min(1, frozen_rms / trainable_rms).detach()`
   so CE cannot drown α separation (soft-collapse mode)
3. CE trains **only** `alpha_byte_proj` / `alpha_length_proj` (+ `obl_gate`)
4. Field dynamics still train; `L_ign = 0`

Checkpoint `save()` / `load()` persist and restore `field_obligatory_hard`.
Unit tests: `test/test_field_obligatory_hard.py`.

## Why prior hard fails chat

Hard-v1 returned **only** frozen random JL α→logits — held probe (~0.86) but
cannot learn language. Soft obligatory re-collapsed under CE (logits→0.992).

## Data / train

| item | value |
|------|-------|
| resume | `checkpoints/atom_native_obligatory_hard/atom_native.pt` @ **2,925,000** |
| chunk +50k | `corpus_dialogue_fr.txt` (non-stream) |
| chunks +100k…+200k | `--stream --data-glob data/dialogue_shards/part_*` (~39 tick/s) |
| hyperparams | hard ON, `L_ign=0`, `--no-episode-reset`, d=64, `field_max_rms=3` |

## Trajectory (new steps from 2.925M → 3.125M = **+200k**)

| stage | total step | logits cos | α cos | chat (honest) |
|-------|------------|------------|-------|---------------|
| hard-v1 end | 2,925,000 | **0.861** | **0.842** | binary noise |
| hard-v2 +50k uncapped | 2,975,000 | **0.996** FAIL | 0.824 | ASCII letter-noise |
| +RMS-cap (no retrain) | 2,975,000 | **0.899** | 0.824 | still noise |
| stream +100k | 3,025,000 | **0.903** | **0.829** | letter fragments |
| stream +150k | 3,075,000 | **0.907** | **0.836** | `'es` / `fea` loops |
| stream +200k | **3,125,000** | **0.916** | **0.842** | short-token loops |

Probe after +200k: logits **≪0.99** (0.916), α alive (0.842). Mild logits drift
0.899→0.916 under continued CE — watch, not soft-collapse (0.99+).

## Chat samples after +200k (best / representative — **not coherent**)

- **Bonjour** → `'…a '…apn '…sp8…a …es 8Dey … 'ea Upa` — Latin scraps, no phrase
- **Qui es-tu ?** → multiline short tokens (`…Ba I…sa` / `'Bs`) — not an answer
- **Il était une fois** → `pac t…pxp pa…a … 8Day…ean… 'ea` — not a story opener
- **Bonjour (deterministic)** → `'ns 'ns '…s 'ps 'ps 'es 'es 'es …` (loop)

**STOP verdict:** after **200k** new hard-v2 steps, generations do **not** show
multi-word readable French. Field probe OK; surface still emits short-token noise.
No fluency claim.

## Artifacts

`docs/artifacts/hard_v2_coherence/{chunk_50k,chunk_100k,chunk_150k,chunk_200k}/`
(train excerpts, probes, chat). Train logs: `logs/hard_v2_chunk_*.txt`.
