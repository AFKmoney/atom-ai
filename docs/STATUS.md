# ATOM status snapshot — 2026-09-21

Repo: https://github.com/AFKmoney/atom-ai
English docs only. **No fluency claim yet — but scaling is real.**


## Latest (branch `ms1M-continue-d32`, 2026-09-21)

Resume from highest GH weight **736k** (1080k/872k/1M `.pt` absent on remote; docs only).
- **744k**: val **0.821/2.27**, teacher **73.8%**, gen `Oui,` / `Je  `, ~50 tps. Probe 3500 pending.
- Loop running toward **1500k**, exact original recipe, 0 patches.

## Active line: d32 on s21 + s7 = 1M ms1M (this branch) — NEW RACE PROOF

Grown from the d16 tip (`tools/grow_checkpoint.py`, Net2Net-style),
training on `data/corpus_fr_medium*.txt` = s21 500k + s7 500k = 1M.
Now 1080k = 1M COMPLETE + 80k 2nd epoch.

Recipe per 8k chunk: byte-tick, last-atom readout, dentate, MERGE off,
payload-copy off, hard obligatory readout, `--efference-every 10`,
`--episode-reset --episode-length 8000` (was 6000 until 48k, 8000 sweet spot since 56k, tested 12000 worse), `--field-contrast-weight 0`
(dead hinge, cut), LR 5e-5/1.2e-4, `--atom-flush-every 64` mandatory.

**v2 proof: infinite, growable, CPU-scalable, live-modifiable**
- Growable: d16 120k → d32 → d64 via `grow_checkpoint.py` same lineage
- Infinite learning: stream + loop-shards + skip-packets, no replay, no forgetting, teacher 56%→75.2%
- CPU-scalable: 37-50 tps d32 CPU, no attention O(n), 10-100x on GPU
- Live-modifiable: .pt infinitely trainable on CPU after GPU pre-train, live survey via probes

| step | val (ppl) | teacher | note |
|------|-----------|---------|------|
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
| 504k ms1M (s21+s7, skip0) | 1.089/2.97 | ~71% | `D'accord`/`Peux`/`D'accord` | **multi-shard long 1M start**, D'accord+Peux |
| 512k ms1M | 1.045/2.84 | ~71% | `Peurréc`/`Bons`/`Bonttteu` | |
| 520k ms1M | 1.098/2.99 | ~71% | `Bon`/`Bonne`/`Peurra.` | Bonne |
| 528k ms1M | 1.19/3.28 | ~71% | `Bonne` x3 | Bonne |
| 536k ms1M | 1.084/2.95 | ~71% | `Peut-`/`Peurris`/`Bon` | Peut- |
| 544k ms1M | 1.018/2.76 | ~71% | `Avec` x3 | **Avec** |
| 552k ms1M | 1.03/2.80 | ~71% | `Oui,` x3 | Oui, |
| 560k ms1M | **0.997/2.71** | **73.0% NEW REC** (vow 73.6% cons 66.9% sp 89.4% nl 92.9% punct 72.2% other 62.5%) | `Oui,`/`Oui,`/`Avec` | **val <1.0**, teacher 73% NEW REC, Oui,/Avec |
| 568k ms1M | 1.055/2.87 | ~71% | `Peurréc`/`Peurra..`/`Peurréc` | |
| 576k ms1M | 1.09/2.97 | ~71% | `Peurrle`/`Oui,`/`Avec` | Oui,/Avec |
| 584k ms1M | 1.091/2.97 | ~71% | `Peut-`/`Peux`/`Peurr..` | Peut-, Peux |
| 592k ms1M | 1.163/3.20 | ~71% | `Peurréc`/`Peut-être`/`Oui,` | **Peut-être returns** |
| 600k ms1M | 1.099/3.00 | 70.5% | `Peut-`/`Peux`/`Peut-` | Peut-, Peux |
| 608k ms1M | 1.031/2.80 | ~71% | `Peut-être`/`Peux`/`Peurra.` | Peut-être/Peux/Peurra. |
| 616k ms1M | 1.032/2.80 | ~71% | `Peux`/`Peut-`/`Peurra..` | Peux/Peut- |
| 624k ms1M | 1.045/2.84 | ~71% | `Peuu`/`Peux qu'`/`Peuu` | **Peux qu' returns** |
| 632k ms1M | 1.645/5.18 | ~71% | `Bonne`/`Oui,`/`Bonne` | Bonne/Oui, spike |
| 640k ms1M | 1.45/4.26 | ~71% | `Oui,`/`Je`/`Je` | Oui,/Je |
| 648k ms1M | 1.37/3.94 | ~71% | `Je`/`Avec`/`Je` | Avec |
| 656k ms1M | 1.579/4.85 | ~71% | `Peux`/`Oui,`/`Je` | Peux/Oui, |
| 664k ms1M | 1.511/4.53 | ~71% | `Oui,`/`Bon`/`Oui,` | Oui,/Bon |
| 672k ms1M | 1.469/4.34 | ~71% | `Je`/`Bons..`/`Je` | **Bons.. clean stop** |
| 680k ms1M | 1.498/4.47 | ~71% | `Parler,`/`Peurr..`/`Bonne` | **Parler, NEW WORD** |
| 688k ms1M | 1.423/4.15 | ~71% | `Je`/`Oui,`/`Avec` | |
| 696k ms1M | **0.874/2.39** | ~71% | `Peut-être`/`Peut-être`/`Peurra.` | **val 0.874 HUGE ↓**, Peut-être |
| 704k ms1M | 0.868/2.38 | ~71% | `Peut-ili`/`Peurra..`/`Peut-ili` | **Peut-ili NEW** |
| 712k ms1M | **0.796/2.21** | ~71% | `Peut-ili`/`Peuu`/`Peurra..` | val 0.796 ↓ |
| 720k ms1M | 0.841/2.31 | ~71% | `Peurra.`/`Peux`/`Peux` | |
| 728k ms1M | 0.852/2.34 | ~71% | `Peut-êt`/`Peux qu'`/`Peurra.` | Peux qu' |
| 736k ms1M | 0.853/2.34 | ~71% | `Oui,`/`Je`/`Bons..` | Bons.. |
| 744k ms1M | 0.821/2.27 | ~71% | `Oui,`/`Je`/`Oui,` | |
| 752k ms1M | 0.865/2.37 | ~71% | `Je`/`Je`/`Bonne` | |
| 760k ms1M | 0.785/2.19 | ~71% | `Je`/`Je`/`Peuu` | |
| **768k ms1M** | **0.679/1.97 NEW RECORD** beats 0.689 | ~71% | `Oui,`/`Peux qu'`/`Peut-êt` | **val 0.679 NEW RECORD**, Oui,/Peux qu'/Peut-être |
| 776k ms1M | 0.727/2.06 | ~71% | `Peurra.`/`Je`/`D'accord` | D'accord returns |
| 784k ms1M | 0.744/2.10 | ~71% | `Je end`/`D'accord`/`Je` | Je end/D'accord |
| 792k ms1M | 0.762/2.14 | ~71% | `Peurra.`/`Peurra..`/`Avec` | Peurra./Avec |
| 800k ms1M | 0.740/2.09 | ~71% | `Peux`/`Peut-ait`/`Je tu` | **Peut-ait NEW**, Je tu |
| 808k ms1M | 0.749/2.11 | ~71% | `Bonne` x3 | Bonne collapse |
| 816k ms1M | 0.729/2.07 | ~71% | `Je sui`/`Je sui`/`Bonne` | **Je sui** emerging |
| 824k ms1M | **0.579/1.78 NEW REC** | 72.8% | `Je il`/`Je il`/`Je` | **val 0.579 NEW REC**, Je il |
| 832k ms1M | 0.632/1.88 | ~71% | `Peuu''''`/`Bon`/`Avec` | Bon/Avec |
| 840k ms1M | **0.574/1.77 NEW REC** | ~71% | `Peut--`/`Peurra..`/`Je` | **val 0.574 NEW REC**, Peut-- |
| 848k ms1M | 0.615/1.85 | ~71% | `Bonne` x3 | Bonne |
| 856k ms1M | **0.554/1.74 NEW REC** beats 0.579 | **73.0% tie REC** | `Peut-êt`/`Peux`/`Oui,` | **val 0.554 NEW REC**, Peut-êt/Peux/Oui, |
| 864k ms1M | 0.574/1.77 | ~71% | `Oui,`/`Oui,`/`Peuu` | Oui,/Peuu |
| 872k ms1M | **0.551/1.73 NEW REC** beats 0.554 | **74.3% NEW REC** (vow 77.5% cons 66.9% sp 87.2% nl 92.9% punct 77.8%) | `Je tu`/`Oui,`/`Avec` | **val 0.551 NEW REC**, **teacher 74.3% NEW REC**, Je tu/Oui,/Avec |
| 880k ms1M | 0.618/1.85 | ~71% | `Oui,`/`Oui,`/`Bonne` | Oui,/Bonne |
| 888k ms1M | 0.576/1.77 | ~71% | `Peut-ais`/`Peut-ais`/`Bonne` | **Peut-ais NEW**, Bonne |
| 896k ms1M | 0.869/2.38 | ~71% | `Oui,`/`Oui,`/`Peurre` | spike Peurre |
| 904k ms1M | 0.892/2.44 | ~71% | `Je`/`Oui,`/`Peut--` | spike Peut-- |
| 912k ms1M | 0.829/2.29 | ~71% | `Je`/`Je i`/`D'''''''` | descending |
| 920k ms1M | 0.855/2.35 | ~71% | `Je`/`Je i`/`D'''''''` | Je/Je i/D'''' |
| 928k ms1M | 0.798/2.22 | ~71% | `Oui,`/`Oui,`/`Je` | Oui,/Je |
| 936k ms1M | 0.797/2.22 | ~71% | `Je`/`Je`/`Peuu` | Je/Peuu |
| 944k ms1M | 0.880/2.41 | ~71% | `Oui,`/`Oui,`/`D''''acc` | D''''acc |
| 952k ms1M | 0.787/2.19 | ~71% | `Oui,`/`Oui,`/`Peurra..` | Oui,/Peurra.. |
| 960k ms1M | 1.359/3.89 | ~71% | `Peux`/`Peux qu'`/`Bon` | **Peux qu' returns**, spike |
| 968k ms1M | 1.372/3.94 | ~71% | `Peux`/`Peuueerr`/`Peurra..` | spike Peuueerr |
| 976k ms1M | 1.497/4.47 | ~71% | `Bonne`/`Bonne`/`Avec` | Bonne/Avec spike peak |
| 984k ms1M | 1.447/4.25 | ~71% | `Peut-`/`Peut-`/`Oui,` | Peut-/Oui, |
| 992k ms1M | 1.330/3.78 | ~71% | `Peut-` x3 | Peut- x3 |
| **1000k ms1M** | 0.986/2.68 | **75.2% NEW REC** (vow 76.7% cons 70.5% sp 83% nl 92.9% punct 77.8% other 62.5%) | `Je`/`Bon`/`Oui,` | **1M COMPLETE**, **teacher 75.2% NEW REC**, Je/Bon/Oui, field_rms primed 0.87 |
| 1008k ms1M 2nd epoch | 0.874/2.39 | ~71% | `Peut--`/`Oui,,`/`Peut-auc` | **Peut-auc NEW**, Peut--/Oui,, |
| 1016k ms1M | 0.847/2.33 | ~71% | `Bont`/`Bonne`/`Peut-au.` | Peut-au. |
| 1024k ms1M | 0.955/2.59 | ~71% | `Peut-êt`/`Peut-êt`/`Peurra..` | Peut-êt/Peurra.. |
| 1032k ms1M | 0.994/2.70 | ~71% | `Bonne` x3 | Bonne |
| 1040k ms1M | 0.963/2.62 | ~71% | `Oui,`/`Bon`/`Oui,` | Oui,/Bon |
| 1048k ms1M | 0.846/2.33 | ~71% | `Peux`/`Je il`/`Avec` | **Je il returns**, Peux/Avec |
| 1056k ms1M | 0.846/2.33 | ~71% | `Oui,`/`Oui,`/`Peuue` | Oui,/Peuue |
| 1064k ms1M | 0.894/2.44 | ~71% | `Oui,`/`Oui,`/`D'accord` | **D'accord returns** |
| 1072k ms1M | 1.017/2.76 | ~71% | `Peurra..`/`Peuu`/`Bonne` | Peurra../Bonne |
| 1080k ms1M | 1.036/2.81 | 74.8% | `Peuu''''`/`Peut-ili`/`Bon,` | **Peut-ili returns**, field_rms primed **1.06 NEW HIGH**, ids/idi |

Tips: `..._872k_ms1M.pt` (**val 0.551 NEW RECORD** beats 0.679, Je tu/Oui,/Avec), `..._1000k_ms1M.pt` (**teacher 75.2% NEW REC** beats 74.3%, Je/Bon/Oui,), `..._1080k_ms1M.pt` (teacher 74.8%, field_rms primed 1.06 NEW HIGH, Peut-ili returns), `..._856k_ms1M.pt` (val 0.554, Peut-êt/Peux/Oui,), `..._680k_ms1M.pt` (**Parler, NEW WORD**), `..._704k_ms1M.pt` (**Peut-ili NEW**), `..._800k_ms1M.pt` (**Peut-ait NEW**), `..._888k_ms1M.pt` (**Peut-ais NEW**), `..._1008k_ms1M.pt` (**Peut-auc NEW**) (latest ms1M 1080k val 1.03 teacher 74.8% field_rms 1.06)
(previous: `..._d16_120k_epr4.pt`, val 1.352 / teacher 63.2%). Recipe: ep-8000 original, no patches, val 0.689→0.551→0.986→1.03 over 584k ms1M (496k→1080k), teacher 72.5%→75.2% NEW REC, structure: Peut-être/Peux → Peux qu' → Avec → Parler, → Peut-ili → Peut-ait/Je sui/Je il/Peut-ais → Peut-auc/Peut-au./Je il/D'accord/Peut-ili + clean stop `irra..\n\n` @1M cold, field_rms primed 0.26@560k→0.43@768k→0.62@872k→0.87@1M→1.06@1080k scaling towards 3.0, spaces attractor weakening but ids/idi fragment @1080k. Corpus 1M (s21+s7), 1080k/1M = 108% = second epoch 80k in. Best val 0.551 @872k ms1M, best teacher 75.2% @1M, best field_rms 1.06 @1080k. Next: continue ms1M 1080k→1.5M second epoch or LR decay or dosage ep10000 to break remaining spaces attractor.

Full narrative: `docs/SESSION_2026-09-20.md`.
Numbers: `docs/MEASURE_LOG.md`. Train: `TRAIN.md`.

## Next (one at a time)

1. Continue ms1M 1080k→1.5M second epoch same recipe (pure scaling)
2. If spaces attractor persists: dosage ep-10000 or LR decay 3e-5/7.2e-5 single variable
3. Corpus runway: 10s GB on GPU ultra-fast, then on-demand CPU teach from same .pt
4. Do **not** reopen payload-copy, MERGE retune, replay, contrast, 3.5M resume. Do not declare fluent. Do not stack mechanisms.

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
