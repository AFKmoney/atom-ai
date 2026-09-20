# ATOM status snapshot — 2026-09-20

Repo: https://github.com/AFKmoney/atom-ai
English docs only. **No fluency claim.**

## Active line: d32 on s21 (this branch)

Grown from the d16 tip (`tools/grow_checkpoint.py`, Net2Net-style),
training on `data/corpus_fr_medium_s21.txt` (seed 21, 500 kB fresh).

Recipe per 8k chunk: byte-tick, last-atom readout, dentate, MERGE off,
payload-copy off, hard obligatory readout, `--efference-every 10`,
`--episode-reset --episode-length 6000`, `--field-contrast-weight 0`
(dead hinge, cut), LR 5e-5/1.2e-4, `--atom-flush-every 64`.

| s21 step | val (ppl) | teacher | note |
|----------|-----------|---------|------|
| 8k | 1.765 (5.84) | 56.0% | knowledge transfers in 1 chunk |
| 16k | 1.585 (4.88) | 61.3% | primed = d16 best |
| 24k | 1.507 (4.51) | 61.3% | loop phase (attractor, not regression) |
| 32k | 1.490 (4.44) | **63.0%** | ties d16 best |
| 40k | 1.417 (4.13) | **64.7%** | record s21, loop `pre` persists but val↓ teacher↑ |
| 48k | 1.523 (4.59) | **67.5%** | val↑ but teacher record, free-run spaces det / `Pe que vec toi.` sampled |
| 56k ep8000 | 1.479 (4.39) | 64.5% | **both regimes FR** `Tu ?\nAssistant: Tu ?\nAs` primed, cold `Peeur: ?\nAssistant: Pe` — dosage test |
| 64k ep8000 | 1.432 (4.19) | 66.5% | **cold best** `Tu ?\nAssistant: Tu ?\nAs` = d16 112k, primed `plais` loop but FR |
| 72k ep8000 | 1.889 (6.61) | 66.0% | `plaisateur` both regimes FR, val spike (2nd epoch) |
| 80k ep8000 | 1.791 (6.00) | 65.5% | cold `?\nAssistant: Tu ?`, primed `d'acco` fragment (new word) |

Tips: `checkpoints/RELEASE/atom_native_d32_40k_s21.pt` (val rec 1.417) + `atom_native_d32_32k_s21.pt`
+ `checkpoints/byte_tick/atom_native_step_64000_d32.pt` (ep8000 cold best val 1.432)
+ `atom_native_step_80000_d32.pt` (ep8000 `d'acco` fragment)
(previous: `checkpoints/RELEASE/atom_native_d16_120k_epr4.pt`,
val 1.352 / teacher 63.2%). Recipe: ep-8000 restores FR both regimes, val spike at 72k (data loop) but teacher stable, next ep-12000 dosage (75% saturated).

Full narrative: `docs/SESSION_2026-09-20.md`.
Numbers: `docs/MEASURE_LOG.md`. Train: `TRAIN.md`.

## Next (one at a time)

1. Continue d32 32k → 40k+ identical (~6% of s21 seen).
2. If saturated loops persist past ~48k: one-chunk ep-8000 dosage A/B.
3. Corpus runway: s21 remainder, seed-7 tail, then multi-shard.
4. Do **not** reopen payload-copy, MERGE retune, replay, contrast,
   3.5M resume. Do not declare fluent. Do not stack mechanisms.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE ·
stacking multiple new mechanisms in one train · declaring fluent ·
force-push · stray `.pt` in git (release via `checkpoints/RELEASE/` only).

---

# Archive — 2026-09-18 3.5M line (do not mix with byte-tick)

The previous STATUS body (payload-copy / MERGE / 3.508M / logits 0.999)
is kept in git history and in the speech-era docs that falsified those
levers (`docs/ATOM_PAYLOAD_PROD.md`, `docs/ALPHA_LOCAL_COPY.md`,
`docs/LINGUISTIC_SPANS.md`). Do not resume that line for speech.
