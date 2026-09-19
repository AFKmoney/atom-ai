# Chat/probe MERGE-off ingest — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Prior (required)

Multi-atom ring @ `4b4d26b`: `atom_flush_every=0` +15k @ 3.483M→**3.498M**;
train still **atoms=1** (MERGE); logits cos **0.999**; chat still noise.
Falsified: flush=0 alone cannot restore multi-atom diversity under operational
MERGE thr=0.45.

Do **not** reopen `byte_decoder` / new aux losses / MERGE thr retune without
evidence. No multi-day train stall.

## Mechanism (one) — MERGE off on chat/probe ingest only

| piece | detail |
|-------|--------|
| change | `forward_packet(..., merge_enabled=)`; `generate_packets` **defaults False**; probe primes with `merge_enabled=False` |
| train | unchanged: `enable_merge=True`, thr=**0.45** (no flag on `transition_loss` path) |
| chat CLI | `--merge-ingest` opt-in to restore legacy MERGE during priming |
| intent | prompt packets remain **distinct living atoms** for α-gated payload copy-bias |
| banned | byte_decoder reopen; long train; MERGE thr change; attention/HF |

Unit: `test/test_chat_merge_off.py` — merge-off keeps n_atoms = n_packets; forced
MERGE collapses to 1; `model.enable_merge` stays True.

## Measure (no train)

Prefer prove on current ckpt **without** +15k. Checkpoint @ **3,498,000**
(`checkpoints/atom_native_chat/atom_native.pt`).

### Unit

60 passed, 1 skipped (full `test/`). New chat-merge-off tests green.

### Chat (deterministic, merge-off default)

| prompt | atoms after | response (honest) |
|--------|-------------|-------------------|
| Bonjour | **6** | `auiseatiu :  rernup e` scraps — no phrase |
| Qui es-tu ? | **5** | empty |
| Il était une fois | **5** | empty |

Contrast `--merge-ingest` Bonjour: **atoms=1**, similar noise.

**Distinct atoms: YES. Multi-word FR: NO.**

### Probe (merge-off ingest)

| metric | prior (MERGE on) | this (MERGE off) |
|--------|------------------|------------------|
| n_atoms after prompt | 1 | **2–3** |
| n_atoms after 20 gen | 1 | **21–22** |
| logits off-diag cos | 0.9987 | **0.9987** (unchanged ≈0.999) |
| α off-diag cos | 0.616 | **0.616** |
| chat | noise | still noise |

Artifacts: `docs/artifacts/chat_merge_off/`.

## Verdict

**FAIL / STOP** — multi-atom ingest restored on chat/probe (atoms 5–6 / 2–3),
but surface logits stay homogenized (~0.999) and chat still noise. No train
burned. Mechanism half-worked (living ring) but decode still ignores diversity.

Root suspicion: `payload_produce` **mean-pools** all living payload embeds +
global hist → multi-atom ring collapses at the copy-bias stage.

Do **not** auto-stack +15k. Next single hypothesis (one sentence):
**Replace mean-pool / global-hist payload copy with per-atom α-local copy-bias
so distinct living prompt atoms can diversify surface logits below ~0.999.**
