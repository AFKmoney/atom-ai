# ATOM status snapshot — 2026-09-20

Repo: https://github.com/AFKmoney/atom-ai
English docs only. **No fluency claim.**

## Active line: d32 on s21 (this branch)

Grown from the d16 tip (`tools/grow_checkpoint.py`, Net2Net-style),
training on `data/corpus_fr_medium_s21.txt` (seed 21, 500 kB fresh).

Recipe per 8k chunk: byte-tick, last-atom readout, dentate, MERGE off,
payload-copy off, hard obligatory readout, `--efference-every 10`,
`--episode-reset --episode-length 8000` (was 6000 until 48k, 8000 sweet spot since 56k, tested 12000 worse), `--field-contrast-weight 0`
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
| 248k ep8000 | **1.025/2.79** | 66.5% | **NEW rec** beats 1.027, `toi`/`Et toi`, `Je Qu'`/`suis d'` |
| 256k ep8000 | 1.133/3.10 | 68.8% vow 72% rec | `Tu`, `Bon, je` |
| 264k ep8000 | 1.188/3.28 | **70.5%** | **teacher NEW rec** vow 73%, `vais d'accord ?`/`Et toi,` |
| 272k ep8000 | 1.256/3.51 | 69.8% punct 72% rec | `vais d'accord ?`, `Je s.\n\n` |
| 280k ep8000 | 1.264/3.54 | 68.5% punct 72% tie | `Tu`, `: Tu`/`Peur::` |
| 288k ep8000 | 1.112/3.04 | 70.3% | `Tu`, `Oui,\nAs`/`Je Tu` |
| 296k ep8000 | 1.123/3.07 | 68.5% | `Tu`, `Bon vai`/`Bonne` **Bonne** |
| 304k ep8000 | 1.105/3.02 | 69.3% | spaces (attractor returns) | `: toi`/`Je es` |
| 312k ep8000 | 1.147/3.15 | 70.0% vow 74% rec | `soir,` **new word** | `Je saan`/`Oui, toi` **perfect** |
| 320k ep8000 | 1.089/2.97 | 70.3% vow **75% NEW REC** | spaces | `Ouis to`/`Oui,`/`Bon,` |
| 328k ep8000 | **0.753/2.12** | 69.0% | **both perfect** `suis d'accord ?\nAssist` both, `Bonne e` | **HUGE val rec beats 1.025** |
| 336k ep8000 | 0.854/2.35 | 69.5% | spaces | `Avec to`/`Peur::` |
| 344k ep8000 | 0.866/2.38 | 68.5% | spaces | `Bons.\n\n` clean stop |
| 352k ep8000 | 0.837/2.31 | 69.3% punct 75% rec | spaces | `Je`/`Oui,` |
| 360k ep8000 | 0.833/2.30 | 67.7% | spaces | `Peurrdi,` |
| 368k ep8000 | 0.889/2.43 | 69.0% | spaces | `Peurr.\n\n` clean stop |
| 376k ep8000 | 0.880/2.41 | **70.8% NEW REC** | `Je   toi`/`Avec  to` | teacher 70.8% NEW REC |
| 384k ep8000 | 0.893/2.44 | 69.5% | `Je`/`Bon,` | spaces |
| 392k ep8000 | 0.870/2.39 | 69.3% vow 73.6% | `Bon  es`/`Bon,` | Bon es |
| 400k ep8000 | 1.480/4.39 | 70.5% tie rec | `Je tu`/`Peurr:::`/`Bonttteu` | val regression but teacher tie rec |
| 408k ep8000 | 1.435/4.20 | 70.8% tie rec | `Peurre`/`Je`/`Bonne` | Bonne |
| 416k ep8000 | 1.480/4.39 | 68.8% | `D'accord` x2 | **D'accord emergence** |
| 424k ep8000 | 1.389/4.01 | 69.8% | `Peurr.\n\n`/`Je` | Peurr. clean stop |
| 432k ep8000 | 1.427/4.17 | **71.0% NEW REC** | `Bonne` x2/`Je` | teacher 71% NEW REC |
| 440k ep8000 | 1.437/4.21 | 71.0% tie rec vow 74.4% | `Je es`/`Peurr:::`/`Je` | |
| 448k ep8000 | 1.514/4.55 | **71.3% NEW REC** | `Peurr:::` x2/`Peurr.` | teacher 71.3% NEW REC |
| 456k ep8000 | 1.408/4.09 | 70.8% | `Peurrle`/`Je toi` | |
| 464k ep8000 | **0.785/2.19** | 71.3% tie rec | `Ouis`/`Oui,`/`Bonne` | **val 0.785 HUGE ↓**, Oui, Bonne |
| 472k ep8000 | **0.747/2.11 NEW REC** | **72.5% NEW REC** cons 69.3% NEW | `Peurréc`/`Peurr..` | val 0.747 NEW REC beats 0.753, teacher 72.5% NEW, cons 69.3% NEW |
| 480k ep8000 | **0.721/2.06 NEW REC** | 71.3% | `Peurr`/`Je`/`D'accord` | val 0.721 NEW REC, D'accord returns |
| 488k ep8000 | 0.721/2.06 tie | 71.3% | `Bon,`/`Je`/`Oui,` | Bon, Oui, |
| 496k ep8000 | **0.689/1.99 NEW REC** | 72.5% tie rec vow 75.2% NEW | `Peurréc`/`Peurrle`/`Peurr:::` | **val 0.689 RECORD**, vow 75.2% NEW REC |
| 504k ep8000 loop | 0.980/2.66 | 71.5% | `Je` x3 | looped corpus, val ↑ |

Tips: `..._472k_s21_ep8000.pt` (**val 0.747 NEW REC**, **teacher 72.5% NEW REC** cons 69.3% NEW), `..._496k_s21_ep8000.pt` (**val 0.689 RECORD**, **vow 75.2% NEW REC**), `..._504k_s21_ep8000.pt` (latest loop 0.98)
(previous: `..._d16_120k_epr4.pt`, val 1.352 / teacher 63.2%). Recipe: ep-8000 sweet spot, val 1.889→0.689 over 432k, teacher 63→72.5% record, free-run FR with spaces attractor since 336k but teacher stable 69-72.5% → knowledge intact, readout attractor. Corpus s21 100% at 500k looped to 504k, next: multi-shard or LR decay or ep-10000 dosage if spaces persist.

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
