# Per-atom α-local copy-bias — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Prior (required)

Chat/probe MERGE-off @ `4deb9a7`: atoms **6** / probe **2–3→21**; logits still
**~0.999**; chat still noise. Root suspicion: `payload_produce` mean-pool +
global-hist homogenizes multi-atom rings at copy-bias.

Do **not** reopen `byte_decoder` / MERGE thr retune without evidence. No
multi-day train stall.

## Mechanism (one) — per-atom α-local copy-bias

| piece | detail |
|-------|--------|
| replace | mean-pool of payload embeds + global hist copy |
| with | each living atom scored by embed·α-gate + energy; soft pool for MLP; **hard α-local winner** owns copy-bias (own hist + position-local) |
| gain | `softplus(payload_copy_scale) * 4` so single-atom copy stays audible vs frozen α RMS |
| keep | hard ON; chat/probe `merge_enabled=False`; train MERGE thr=**0.45**; flush=0; nxp=0.05 |
| banned | byte_decoder reopen; long train; MERGE thr change; attention/HF |

Unit: `test/test_atom_payload_prod.py` — distinct payload sets → logits cos **<0.99**;
energy flips prefer A vs z bytes; α=0 still zeros; decoder frozen.

## Measure

### Unit (no train)

62 passed, 1 skipped (full `test/`). New α-local tests green.

### Prove on current ckpt (3,498,000) — no train

| metric | MERGE-off prior | α-local (no train) |
|--------|-----------------|--------------------|
| logits off-diag cos | 0.9987 | **0.9988** (unchanged ≈0.999) |
| α off-diag cos | 0.616 | **0.616** |
| chat atoms Bonjour | 6 | **4** (merge-off) |
| multi-word FR | no | **no** (empty / scraps) |

### Tiny fine-tune ≤10k then STOP

| item | value |
|------|-------|
| resume | chat ckpt @ **3,498,000** |
| budget | **+10k** → **3,508,000** (`checkpoints/atom_native_alpha_local/`) |
| hyperparams | hard ON, MERGE ON thr=0.45, flush=0, nxp=0.05, d=64 |
| train atoms | still **1** every step (MERGE) |

| metric | after +10k |
|--------|------------|
| logits off-diag cos | **0.9987** (still ≈0.999) |
| α off-diag cos | **0.709** |
| chat | still noise / empty — **no** multi-word FR |

Artifacts: `docs/artifacts/alpha_local_copy/`.

## Verdict

**FAIL / STOP** — unit proves diversification on tiny random surface; on the
living hard ckpt, frozen α map + capped MLP still dominate surface RMS so
α-local payload copy cannot pull probe logits cos below ~0.99. +10k under
train MERGE (atoms=1) does not fix chat. No further train.

Do **not** auto-stack another +10k/25k. Next single hypothesis (one sentence):
**Shrink or freeze-out the hard frozen α→logit map so living-atom α-local
payload copy is the primary surface under multi-atom merge-off ingest.**
