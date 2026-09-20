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
| 88k ep12000 | 1.842 (6.31) | 64.0% | ep12000 worse: `pen pen` loop, val↑ teacher↓ — ep8000 sweet spot |
| 96k ep8000 | 1.809 (6.11) | 66.0% | retour ep8000: `vec toi ?\nAssistant:` FR |
| 104k ep8000 | 1.706 (5.51) | 67.0% | **both regimes perfect** `Tu ?\nAssistant: Tu ?\nAs` primed |
| 112k ep8000 | 1.645 (5.18) | **69.0%** | teacher record, val ↓ trend, `plainuo`/`Pe ven` |
| 120k ep8000 | 1.716 (5.56) | 66.5% | `Ouisateu`/`Je veur:`/` d'accor` (d'accord returns) |
| 128k ep8000 | 1.677 (5.35) | 67.2% | primed ` Tu plus.\n\n` clean stop |
| 136k ep8000 | **1.370 (3.93)** | 68.2% | **val rec s21**, cold+primed perfect `Tu ?\nAssistant: Tu ?\n` |
| 144k ep8000 | **1.365 (3.92)** | 66.2% | **val rec**, both perfect `Tu ?\nAssistant: Tu ?\nAs`, `Parle  v` |
| 152k ep8000 | 1.477 (4.38) | 67.5% | primed ` D'accord ?\nAssistant: D` full D'accord |
| 160k ep8000 | **1.346 (3.84)** | 67.0% | **val rec beats d16 1.352**, d32 > d16, `Parle  q`/`Je ve` |
| 168k ep8000 | 1.364 (3.91) | 68.0% | cold `Tu veur:`, `Parle  ?`/`Je qu'e` |
| 176k ep8000 | 1.383 (3.99) | 68.0% | `Peur:`, `?\nAssistant: Tu` |
| 184k ep8000 | 1.399 (4.05) | 67.0% | primed `D'accord ?` returns, `Je toi,` |
| 192k ep8000 | 1.423 (4.15) | 67.7% | cold `vais d'accord ?`, primed `Ouis d'accord ?`, `Parle Q` |
| 200k ep8000 | **1.063 (2.90)** | 67.2% | **HUGE rec**, primed ` Oui,  Et toi ?\nAssistan` best FR, `Je vais`/`Je vaiss` |
| 208k ep8000 | **1.055 (2.87)** | 68.0% | **NEW rec** beats 1.063, cold `vec vais d'accord ?`, `Bon Qu'`/`Oui ?\nAs` |
| 216k ep8000 | 1.086 (2.96) | **69.0%** | teacher tie rec, cold `toi ?\nAssistant: toi` perfect |
| 224k ep8000 | 1.077 (2.94) | 66.5% | `Tu toi`, `Bonnt to`/`Peur: t`/`Bon moi` |
| 232k ep8000 | **1.027 (2.79)** | 68.2% | **NEW rec** beats 1.055, cold ` toi.\n\n` clean stop, `Je s.\n\n` |
| 240k ep8000 | 1.042 (2.84) | 67.7% | `?\nAssistant: Tu`, `Bon ?\nA`/`Bon Tu`/`sesta` |

Tips: `..._40k_s21.pt` (val 1.417) + `..._32k_s21.pt`
+ `..._160k_s21_ep8000.pt` (**val 1.346 beats d16 1.352**)
+ `..._200k_s21_ep8000.pt` (**val 1.063 HUGE**, primed `Oui, Et toi ?` best FR, `Je vais`)
+ `..._208k_s21_ep8000.pt` (**val 1.055 NEW rec** ppl 2.87)
+ `..._232k_s21_ep8000.pt` (**val 1.027 NEW rec** ppl 2.79 beats all, `toi.\n\n` clean stop)
+ `..._240k_s21_ep8000.pt` (latest, 1.042)
(previous: `..._d16_120k_epr4.pt`, val 1.352 / teacher 63.2%). Recipe: ep-8000 sweet spot, val 1.889→1.027 over 160k, teacher 63→69% stable, free-run FR constant with new comps `vais d'accord`, `Oui, Et toi ?`, `Je vais`, `Parle`, `Je s.\n\n`. Corpus s21 ~48% at 240k, runway remains. Next: continue ep-8000 identical 240k→248k.

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
