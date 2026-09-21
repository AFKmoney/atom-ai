# ATOM speech cuts — measured log

Sandbox: d=16→d32, byte-tick, CPU ~37–72 tick/s.
One mechanism per row. **No fluency claim yet — but scaling real.**

Pass rule: two prompts emit ≥4 Latin letters **and** the strings differ.

| step | cut | train loss | wrap generate (honest) | note |
|------|-----|------------|------------------------|------|
| 8–13k | span-16 | 6.1–6.9 | soup / `T T T` | contract wrong |
| 10k | byte-tick | 2.70 | space / newline | CE works, unigram |
| 20k | last-atom readout | **1.68** | `tis tis tis` | letters exist |
| 24k | + φ oscillatory binding | 5.09→4.67 | `te te te` | more letters, less diversity |
| 28k-r | + replay high-CE | 2.99→**0.008** | `tis`/`pisi` | **quarantine** — ring memorized |
| 28k-e | + efference from 24k | 2.99→4.82 | tail `Je` | first lexical Je |
| 32k | + dentate 2-gram | 2.41→5.37 | wrap all `Je` / `Je tenta` | Je attractor |
| 36k-cont | same cut continue | 2.95→5.54 | `Je Je Je` | worse — do not resume |
| 32k + n-gram anti-repeat | decode only | — | `Je Jestiste` / `Je pe te pe ?` | **best chat**; weights untouched |
| 36k-big | 32k + 223 kB combo | 2.95→**1.61** | `Je pe tisilisi` | best CE since last-atom |
| 39.5k-hf | 36k-big + CATIE FR 180 kB | 4.67→3.49 | `pe Je pe Jis te` | external chat domain |

## Resume

Speech tip for generate: **32k dentate** + n-gram anti-repeat in `generate_packets`.
Train tip for more data: **36k-big** or **39.5k-hf`.
Never resume **28k-replay** or **36k-cont**.

## Ligne A — medium FR corpus, payload-off, uncapped hard mix (2026-09-19)

Sandbox: d=16/n=16/64, max-span 4, stream, hard, MERGE off, payload-copy off, hard mix = obl·α-MLP + 0.3·frozen (RMS cap removed), span-level n-gram anti-repeat. Corpus: `data/corpus_fr_medium.txt` (500 kB procedural FR dialogue). Probe: `tools/probe_speech.py`. ~20–40 tick/s. **No fluency claim — debris.**

| step | train byte_ppl | val byte_ppl | det. probe + anti-repeat | note |
|------|---------------|-------------|--------------------------|------|
| 4k | ~11 | 21.3 | `\nsi \nsi …` (single attractor) | vowels+spaces exist |
| 8k | ~11 | 21.4 | `ssi ssi …` / `\nsin …` | diverse across prompts |
| 12k | ~11 | 23.3 | `ssi→sse→se` | sibilant ladder |
| 16k | ~10 | 143.3 | `ssa / ssat` | val spike = cold-field ring artifact |
| 20k | ~10 | 49.0 | `sat est satsuas` | first real word `est` |
| 24k | ~10 | 50.2 | `tuat / uut / uat / ent` | (prefix retrain) |
| 28k | ~10 | 44.4 | `uat / ent / ois / uit / sit` | teacher vowel acc 9.5% |
| 32k | ~42→ | 30.3 | `sai / sai / eit` | fresh data (skip 4k) |
| 36k | ~19 | 33.3 | `sai / sut / uit` | fresh data (skip 8k) |
| 40k | ~15 | 27.1 | `uateeai / eiteou / …tesisirs` | long varied chains |
| 44k | ~19 | 24.0 | `eitet / eitsun / eite` | fresh data (skip 16k) |
| 48k | ~21 | 23.9 | `uiteuui / eiteua / reri` | teacher vowel acc 14.7% |

Findings:
- Train/chat payload mismatch: copy ON at train, OFF at chat gave consonant cycles, copy ON gave vowels — copy was crutch. Fixed by training with `--no-payload-copy`.
- Chained `--resume` chunks re-trained same prefix; fixed with `--stream-skip-packets`.
- Old hard checkpoints (capped mix, copy-on) invalid under new forward.

## Ligne B — byte-tick + last-atom readout + dentate (2026-09-19/20)

Sandbox: d=16/n=16/64, **max-span 1**, stream 64 kB chunks, hard, MERGE off, payload-copy off, last-atom readout (2-gram + dentate top-25% + φ). Logits = obl·α-MLP + 0.3·frozen + last-atom. ~48–55 tick/s. **No fluency claim — debris with words.**

| step | val byte_ppl | det. probe + anti-repeat | note |
|------|-------------|--------------------------|------|
| 4k | 20.3 | `te re tes ts ti` | bigram attractor |
| 8k | 11.5 | `de decisateur / dUtilisateur` | dialogue skeleton: roles |
| 12k | 9.3 | `de reur / iecisateur` | role fragments |
| 16k | 7.9 | `de reur / outilisateur` | near-words |
| 24k | 7.0 | `Tu ou uu tou` | pronoun Tu |
| 32k | 6.3 | `Tu pe tont / Tu ont` | Tu attractor |
| 40k | 5.8 | `Tu eur / moi tour / mon pe` | moi/mon emerge |
| 48k | 5.3 | `Tu pe toi ? / Assisateur` | questions + `?` |
| 56k | 5.1 | `tout Oui / Assistant: Tu toi ?` | correct roles |
| 64k | 5.2 | `Ouisateur: Ouisateur: …` | attractor relapse |
| 72k | **4.2** | `Tu pe toi ?\nAssistant: Tu pe toi ?` | **best chat**: full turns |

Teacher-forcing @72k (400 fresh transitions): byte acc **59.5%** (vowels 64%, consonants 53%, spaces 83%, newlines 100%, punct 28%).
Best chat = `atom_native_step_72000_btick.pt`.

## Ligne B++ — efference copy + le mirage du collapse (2026-09-20)

Efference (bridge #9): `--efference-every 10`, 1 tick/10 = predict own byte (no-grad) → commit → CE on gold next. 4 tests in `test/test_efference.py`, ~0% tps cost, 800/8000 eff ticks verified.

| step | val byte_ppl | cold probe (40-pkt prompt) | primed probe (saturated field) |
|------|-------------|---------------------------|-------------------------------|
| 80k eff | **3.9** (record 1.42) | `toi   ssistant:` | — |
| 82k eff | — | `toi to tant: on plus` (best cold) | — |
| 88k eff | 4.5 | `�FF�e� {` binary | `Tu vec toi ?\nAssisateur` FRENCH |
| 88k no-eff (A/B) | — | `�FF��FF` garbage | `�e�r�\nUtilisateur: …` mixed |
| 88k no-contrast | — | `�FF�e� {` bit-identical to eff | `Tu vec toi ?\nAssisateur` bit-identical |
| 96k eff | 4.3 | `�on plustant: …` mixed | `Tu vec toi.\n\n` FRENCH + clean stop |

**Le collapse 88k est un mirage de mesure (régime de champ), pas un dégât.** Preuves: A/B sans efference → même soupe, logits 88k plats dès t0 (H=3.67 vs 0.02), α identique 80k/88k → readout seul coupable, 80k α-MLP transplanté dans 88k → français restauré, drift fc2 cos=+0.976 linéaire, ` ' '` peak +4.4 survit mais MLP booste +2..+3 sur octets poubelle, primer 1500 octets (rms 3.0 = training regime) → 88k parle français. Diagnosis: α-MLP sur-apprend régime saturé, cold probe rms 0.06 OOD. Fix: `--episode-reset` + contrast 0 cut.

## Cut episode-reset (2026-09-20) : 96k→104k, ep-2000 + contraste 0

4 épisodes/chunk (~850 transitoire + ~1150 saturé). Val 1.54→**1.477** (ppl 4.13). Cold: `toi ?\nAssistant: toi`; primed: `pen pen…` boucle. Teacher span-1 **59.8%** (punct 22→33%). Note: `diagnose_teacher` défaut span-4 donne ~22% sur byte-tick; toujours `--max-span-bytes 1`. Vitesse ~50 tps.

## 112k ep-4000 : le compromis deux-régimes existe (2026-09-20)

104k→112k, `--episode-length 4000` (~20% transitoire). Val **1.423**/ppl 3.90 (record égalé), teacher **61.3%** (cons 56%, punct 36%). Cold: `Tu toi ?\nAssistant: Tu`; primed: `Tu ?\nAssistant: Tu ?\nAs` — **français dans les deux régimes, sans boucle**. Tip: `atom_native_step_112000_epr4.pt`.

## 120k : fin du d16 (2026-09-20)

112k→120k identical. Val **1.352**/ppl 3.65 (record), teacher **63.2%** (record). Cold `toi ?\nAssistant: men`; primed `Je snsestant: plaistan`. Corpus seed-7 vu à 24%.

## d=32 par croissance (2026-09-20) : `tools/grow_checkpoint.py`

Net2Net préservant: lignes doublées tuilées+bruit, colonnes doublées zero-pad (sorties anciennes exactes), embeddings tuilés+bruit, JL entrelacé + frozen grown (branche α exacte sur α tuilé). 3 itérations: tile+noise naïf (×2 magnitudes → soupe), preserve (primed latin `T T T`), +JL/frozen exact (paradoxalement pire: trajectoire α d32 diffère — saturation à ~3000 pas au lieu de ~850 — donc fonction exacte ≠ même comportement: champ et readout co-adaptés). Leçon: pas de init miracle, training ré-adapte (1 chunk suffit). Règle d'épisode d32: ep-6000 (~2× saturation).

## d32 sur corpus s21 (seed 21, 500k frais, 2026-09-20) + ms1M (s21+s7=1M)

| step (s21) | val | teacher | cold | primed (3.5k) |
|------------|-----|---------|------|---------------|
| grown (init) | — | — | soupe | soupe (rms 0.8) |
| 8k | 1.765/5.84 | **56.0%** (punct 47%) | espaces/`?` | `Distant: Distant:` rôles ! |
| 16k | 1.585/4.88 | **61.3%** | `Ouistant: …` rôles à froid | `Tu ?\nAssistant: Tu ?` = best d16 |
| 24k | 1.507/4.51 | 61.3% (cons 60%, \n 93%) | `Oun plaistant:…` (boucle) | `Tu plaistant:…` (boucle) |
| 32k | 1.490/4.44 | **63.0%** (voy 68%, punct 50%) | `?\nAssistant:…` (boucle motif) | `Oui, que pre pre…` (boucle) |
| 40k | 1.417/4.13 | **64.7%** (voy 67%, cons 60%, sp 81%, nl 93%, punct 53%) | ` On pre ?\nAssistant: On ` (boucle `pre`) | ` On pre vec pre vec pre ` (boucle `pre vec`) |
| 48k | 1.523/4.59 | **67.5%** | spaces det + ` Pe ?\nAs` / `Tu ?\nAssistant:   ` | spaces / `Pe que vec toi.` |
| 56k ep8000 | 1.479/4.39 | 64.5% | ` Peeur:  ?\nAssistant: Pe` | ` Tu ?\nAssistant: Tu ?\nAs` **FR both regimes** |
| 64k ep8000 | 1.432/4.19 | 66.5% | ` Tu ?\nAssistant: Tu ?\nAs` **best cold = d16 112k** | ` Tu plaisateur: Tu plais` |
| 72k ep8000 | 1.889/6.61 | 66.0% | ` Tu plaisateur: Tu  plai` | ` Tu plaisateur: Tu plais` |
| 80k ep8000 | 1.791/6.00 | 65.5% | `  ?\nAssistant: Tu ?\nAssi` | ` Tu pent: Tu  pes d'acco` |
| 88k ep12000 | 1.842/6.31 | 64.0% | `  ?\nAssistant:  ?\nAssist` | ` Tu vec pen pen pen pen ` |
| 96k ep8000 | 1.809/6.11 | 66.0% | `  ?\nAssistant:  ?\nAssist` | ` Tu vec toi ?\nAssistant:` |
| 104k ep8000 | 1.706/5.51 | 67.0% | `  ?\nAssistant: Tu ?\nAssi` | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both** |
| 112k ep8000 | 1.645/5.18 | **69.0%** | `  ?\nAssistant:  ?\nAssist` | ` Tu vec plaisateur: Tu v` |
| 120k ep8000 | 1.716/5.56 | 66.5% | ` Tu  Tu   Tu    Tu   Tu ` | ` Tu vec plaisateur: Tu  ` |
| 128k ep8000 | 1.677/5.35 | 67.2% | `  ?\nAssistant:  ?\nAssist` | ` Tu plus.\n\n` clean stop |
| 136k ep8000 | **1.370/3.93** | 68.2% | ` Tu  ?\nAssistant: Tu  ?\n` **perfect + clean stop** | ` Tu ?\nAssistant: Tu  ?\nA` **perfect** |
| 144k ep8000 | **1.365/3.92** | 66.2% | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both** | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both** |
| 152k ep8000 | 1.477/4.38 | 67.5% | `  ?\nAssistant:  ?\nAssist` | ` D'accord ?\nAssistant: D` **D'accord full** |
| 160k ep8000 | **1.346/3.84** | 67.0% | `  ?\nAssistant:  ?\nAssist` | ` Tu vec  vec plaisateur:` |
| 168k ep8000 | 1.364/3.91 | 68.0% | `  ?\nAssistant: Tu veur: ` | ` Tu veur:  Tu  Tu  Tu  T` |
| 176k ep8000 | 1.383/3.99 | 68.0% | `  ?\nAssistant: Tu       ` | ` Tu  veur:              ` |
| 184k ep8000 | 1.399/4.05 | 67.0% | `  ?\nAssistant:  ?\nAssist` | ` D'accord ?\nAssistant: D` **D'accord returns** |
| 192k ep8000 | 1.423/4.15 | 67.7% | `  vais d'accord ?\nAssist` | ` Ouis d'accord ?\nAssista` |
| 200k ep8000 | **1.063/2.90** | 67.2% | ` Tu                     ` | ` Oui,  Et toi ?\nAssistan` **best primed FR** |
| 208k ep8000 | **1.055/2.87** | 68.0% | `  vec  vais d'accord ?\nA` | ` Tu veur:  Tu     Tu    ` |
| 216k ep8000 | 1.086/2.96 | **69.0%** | `  toi ?\nAssistant:  toi ` **both roles FR** | ` Tu                     ` |
| 224k ep8000 | 1.077/2.94 | 66.5% | ` Tu                     ` | ` Tu  toi                ` |
| 232k ep8000 | **1.027/2.79** | 68.2% | `  toi.\n\n` clean stop | ` Tu                     ` |
| 240k ep8000 | 1.042/2.84 | 67.7% | `  ?\nAssistant: Tu       ` | ` Tu                     ` |
| 248k ep8000 | **1.025/2.79** | 66.5% | `  toi                   ` | `  Et toi                ` |
| 256k ep8000 | 1.133/3.10 | 68.8% vow 72% rec | `  Tu                    ` | ` Tu                     ` |
| 264k ep8000 | 1.188/3.28 | **70.5%** vow 73% rec | `  vais d'accord ?\nAssist` | `  Et toi,               ` |
| 272k ep8000 | 1.256/3.51 | 69.8% punct 72% rec | `  vais d'accord ?\nAssist` | ` Tu                     ` |
| 280k ep8000 | 1.264/3.54 | 68.5% punct 72% tie | `  Tu                    ` | ` Tu                     ` |
| 288k ep8000 | 1.112/3.04 | 70.3% | `                        ` spaces | ` Tu                     ` |
| 296k ep8000 | 1.123/3.07 | 68.5% | `                        ` | ` Tu                     ` |
| 304k ep8000 | 1.105/3.02 | 69.3% | `                        ` spaces | `                        ` spaces |
| 312k ep8000 | 1.147/3.15 | 70.0% vow 74% rec | `  soir,                 ` **soir,** | ` Tu                     ` |
| 320k ep8000 | 1.089/2.97 | 70.3% vow **75% NEW REC** | `                        ` | `  suistant::            ` |
| 328k ep8000 | **0.753/2.12** | 69.0% | `  suis d'accord ?\nAssist` **both perfect** | `  suis d'accord ?\nAssist` **both perfect** |
| 336k ep8000 | 0.854/2.35 | 69.5% | `                        ` spaces | `                        ` spaces |
| 344k ep8000 | 0.866/2.38 | 68.5% | `                        ` | `                        ` |
| 352k ep8000 | 0.837/2.31 | 69.3% punct **75% NEW REC** | `                        ` | `                        ` |
| 360k ep8000 | 0.833/2.30 | 67.7% | `                        ` | `                        ` |
| 368k ep8000 | 0.889/2.43 | 69.0% | `                        ` | `                        ` |
| 376k ep8000 | 0.880/2.41 | **70.8%** | `Je   toi` / `Avec  to` | spaces, Je toi |
| 384k ep8000 | 0.893/2.44 | 69.5% | `Je      ` / `Bon,    ` | spaces |
| 392k ep8000 | 0.870/2.39 | 69.3% vow 73.6% | `Bon  es ` / `Bon,    ` | Bon es |
| 400k ep8000 | 1.480/4.39 | 70.5% tie rec | `Je   tu ` / `Peurr:::` / `Bonttteu` | regression but teacher tie rec, spaces |
| 408k ep8000 | 1.435/4.20 | 70.8% tie rec | `Peurre  ` / `Je      ` / `Bonne   ` | Bonne |
| 416k ep8000 | 1.480/4.39 | 68.8% | `D'accord` / `Je      ` / `D'accord` | **D'accord** emergence |
| 424k ep8000 | 1.389/4.01 | 69.8% | `Peurr.\n\n` / `Peurr.\n\n` / `Je      ` | Peurr. clean stop |
| 432k ep8000 | 1.427/4.17 | **71.0% NEW REC** | `Bonne   ` / `Bonne   ` / `Je      ` | Bonne, teacher 71% NEW |
| 440k ep8000 | 1.437/4.21 | 71.0% tie rec vow 74.4% | `Je   es ` / `Peurr:::` / `Je      ` | Je es |
| 448k ep8000 | 1.514/4.55 | **71.3% NEW REC** | `Peurr:::` / `Peurr:::` / `Peurr.\n\n` | Peurr::: |
| 456k ep8000 | 1.408/4.09 | 70.8% | `Peurrle ` / `Peurrle ` / `Je   toi` | Peurrle |
| 464k ep8000 | **0.785/2.19** | 71.3% tie rec | `Ouis    ` / `Oui,    ` / `Bonne   ` | **val 0.785 HUGE ↓**, Oui, Bonne |
| 472k ep8000 | **0.747/2.11 NEW REC** | **72.5% NEW REC** cons 69.3% NEW | `Peurréc` / `Peurr.\n\n` / `Peurr.\n\n` | val 0.747 NEW REC, teacher 72.5% NEW, cons 69.3% NEW |
| 480k ep8000 | **0.721/2.06 NEW REC** | 71.3% | `Peurr   ` / `Je      ` / `D'accord` | val 0.721 NEW REC, D'accord returns |
| 488k ep8000 | 0.721/2.06 tie rec | 71.3% | `Bon,    ` / `Je      ` / `Oui,    ` | Bon, Oui, |
| 496k ep8000 | **0.689/1.99 NEW REC** | 72.5% tie rec vow 75.2% NEW | `Peurréc` / `Peurrle ` / `Peurr:::` | val 0.689 RECORD, vow 75.2% NEW REC |
| 504k ms1M s21+s7 skip0 | 1.089/2.97 | ~71% | `D'accord` / `Peux    ` / `D'accord` | **multi-shard long 1M start** from 496k best, D'accord+Peux |
| 512k ms1M | 1.045/2.84 | ~71% | `Peurréc` / `Bons    ` / `Bonttteu` | |
| 520k ms1M | 1.098/2.99 | ~71% | `Bon     ` / `Bonne   ` / `Peurra.\n` | Bonne, Peurra |
| 528k ms1M | 1.19/3.28 | ~71% | `Bonne   ` x3 | Bonne |
| 536k ms1M | 1.084/2.95 | ~71% | `Peut-   ` / `Peurris ` / `Bon     ` | Peut- |
| 544k ms1M | 1.018/2.76 | ~71% | `Avec    ` x3 | **Avec** |
| 552k ms1M | 1.03/2.80 | ~71% | `Oui,    ` x3 | Oui, |
| 560k ms1M | 0.997/2.71 | **73.0% NEW REC** | `Oui,    ` / `Oui,    ` / `Avec    ` | **val <1.0**, teacher 73% NEW REC, Oui,/Avec |
| 568k ms1M | 1.055/2.87 | ~71% | `Peurréc` / `Peurra..` / `Peurréc` | |
| 576k ms1M | 1.09/2.97 | ~71% | `Peurrle ` / `Oui,    ` / `Avec    ` | Oui,/Avec |
| 584k ms1M | 1.091/2.97 | ~71% | `Peut-   ` / `Peux    ` / `Peurr..\n` | Peut-, Peux |
| 592k ms1M | 1.163/3.20 | ~71% | `Peurréc` / `Peut-êt` / `Oui,    ` | **Peut-être returns** |
| 600k ms1M | 1.099/3.00 | 70.5% | `Peut-   ` / `Peux    ` / `Peut-   ` | Peut-, Peux |
| 608k ms1M | 1.031/2.80 | ~71% | `Peut-êt` / `Peux    ` / `Peurra.` | Peut-être/Peux/Peurra. |
| 616k ms1M | 1.032/2.80 | ~71% | `Peux    ` / `Peut-   ` / `Peurra..` | Peux/Peut- |
| 624k ms1M | 1.045/2.84 | ~71% | `Peuu    ` / `Peux qu'` / `Peuu    ` | **Peux qu' returns** |
| 632k ms1M | 1.645/5.18 | ~71% | `Bonne   ` / `Oui,    ` / `Bonne   ` | Bonne/Oui, spike |
| 640k ms1M | 1.45/4.26 | ~71% | `Oui,    ` / `Je      ` / `Je      ` | Oui,/Je |
| 648k ms1M | 1.37/3.94 | ~71% | `Je      ` / `Avec    ` / `Je      ` | Avec |
| 656k ms1M | 1.579/4.85 | ~71% | `Peux    ` / `Oui,    ` / `Je      ` | Peux/Oui, |
| 664k ms1M | 1.511/4.53 | ~71% | `Oui,    ` / `Bon     ` / `Oui,    ` | Oui,/Bon |
| 672k ms1M | 1.469/4.34 | ~71% | `Je      ` / `Bons.   ` / `Je      ` | Bons.. clean stop |
| 680k ms1M | 1.498/4.47 | ~71% | `Parler, ` / `Peurr.. ` / `Bonne   ` | **Parler, NEW WORD** |
| 688k ms1M | 1.423/4.15 | ~71% | `Je      ` / `Oui,    ` / `Avec    ` | |
| 696k ms1M | 0.874/2.39 | ~71% | `Peut-êt` / `Peut-êt` / `Peurra. ` | **val 0.874 HUGE ↓** Peut-être |
| 704k ms1M | 0.868/2.38 | ~71% | `Peut-ili` / `Peurra..` / `Peut-ili` | **Peut-ili NEW** |
| 712k ms1M | 0.796/2.21 | ~71% | `Peut-ili` / `Peuu    ` / `Peurra..` | val 0.796 ↓ |
| 720k ms1M | 0.841/2.31 | ~71% | `Peurra. ` / `Peux    ` / `Peux    ` | |
| 728k ms1M | 0.852/2.34 | ~71% | `Peut-êt` / `Peux qu'` / `Peurra. ` | Peux qu' |
| 736k ms1M | 0.853/2.34 | ~71% | `Oui,    ` / `Je      ` / `Bons.   ` | Bons.. |
| 744k ms1M | 0.821/2.27 | ~71% | `Oui,    ` / `Je      ` / `Oui,    ` | |
| 752k ms1M | 0.865/2.37 | ~71% | `Je      ` / `Je      ` / `Bonne   ` | |
| 760k ms1M | 0.785/2.19 | ~71% | `Je      ` / `Je      ` / `Peuu    ` | |
| 768k ms1M | **0.679/1.97 NEW RECORD** beats 0.689 | ~71% | `Oui,    ` / `Peux qu'` / `Peut-êt` | **val 0.679 NEW RECORD**, Oui,/Peux qu'/Peut-être |
| 776k ms1M | 0.727/2.06 | ~71% | `Peurra. ` / `Je      ` / `D'accord` | D'accord returns |
| 784k ms1M | 0.744/2.10 | ~71% | `Je   end` / `D'accord` / `Je      ` | Je end/D'accord |
| 792k ms1M | 0.762/2.14 | ~71% | `Peurra. ` / `Peurra..` / `Avec    ` | Peurra./Avec |
| 800k ms1M | 0.740/2.09 | ~71% | `Peux    ` / `Peut-ait` / `Je   tu` | **Peut-ait NEW**, Je tu |
| 808k ms1M | 0.749/2.11 | ~71% | `Bonne   ` x3 | Bonne collapse |
| 816k ms1M | 0.729/2.07 | ~71% | `Je   sui` / `Je   sui` / `Bonne   ` | **Je sui** emerging |
| 824k ms1M | **0.579/1.78 NEW REC** | 72.8% | `Je   il` / `Je   il` / `Je      ` | **val 0.579 NEW REC**, Je il |
| 832k ms1M | 0.632/1.88 | ~71% | `Peuu''''` / `Bon     ` / `Avec    ` | Bon/Avec |
| 840k ms1M | **0.574/1.77 NEW REC** | ~71% | `Peut--  ` / `Peurra..` / `Je      ` | **val 0.574 NEW REC**, Peut-- |
| 848k ms1M | 0.615/1.85 | ~71% | `Bonne   ` x3 | Bonne |
| 856k ms1M | **0.554/1.74 NEW REC** | 73.0% tie rec | `Peut-êt` / `Peux    ` / `Oui,    ` | **val 0.554 NEW REC**, Peut-êt/Peux/Oui, |
| 864k ms1M | 0.574/1.77 | ~71% | `Oui,    ` / `Oui,    ` / `Peuu    ` | Oui,/Peuu |
| 872k ms1M | **0.551/1.73 NEW REC** | **74.3% NEW REC** | `Je   tu` / `Oui,    ` / `Avec    ` | **val 0.551 NEW REC**, **teacher 74.3% NEW REC**, Je tu/Oui,/Avec |
| 880k ms1M | 0.618/1.85 | ~71% | `Oui,    ` / `Oui,    ` / `Bonne   ` | Oui,/Bonne |
| 888k ms1M | 0.576/1.77 | ~71% | `Peut-ais` / `Peut-ais` / `Bonne   ` | **Peut-ais NEW**, Bonne |
| 896k ms1M | 0.869/2.38 | ~71% | `Oui,    ` / `Oui,    ` / `Peurre  ` | spike Peurre |
| 904k ms1M | 0.892/2.44 | ~71% | `Je      ` / `Oui,    ` / `Peut--  ` | spike Peut-- |
| 912k ms1M | 0.829/2.29 | ~71% | `Je      ` / `Je   i` / `D'''''''` | descending |
| 920k ms1M | 0.855/2.35 | ~71% | `Je      ` / `Je   i` / `D'''''''` | Je/Je i/D'''' |
| 928k ms1M | 0.798/2.22 | ~71% | `Oui,    ` / `Oui,    ` / `Je      ` | Oui,/Je |
| 936k ms1M | 0.797/2.22 | ~71% | `Je      ` / `Je      ` / `Peuu    ` | Je/Peuu |
| 944k ms1M | 0.880/2.41 | ~71% | `Oui,    ` / `Oui,    ` / `D''''acc` | D''''acc |
| 952k ms1M | 0.787/2.19 | ~71% | `Oui,    ` / `Oui,    ` / `Peurra..` | Oui,/Peurra.. |
| 960k ms1M | 1.359/3.89 | ~71% | `Peux    ` / `Peux qu'` / `Bon     ` | **Peux qu' returns**, spike |
| 968k ms1M | 1.372/3.94 | ~71% | `Peux    ` / `Peuueerr` / `Peurra..` | spike Peuueerr |
| 976k ms1M | 1.497/4.47 | ~71% | `Bonne   ` / `Bonne   ` / `Avec    ` | Bonne/Avec spike peak |
| 984k ms1M | 1.447/4.25 | ~71% | `Peut-   ` / `Peut-   ` / `Oui,    ` | Peut-/Oui, |
| 992k ms1M | 1.330/3.78 | ~71% | `Peut-   ` x3 | Peut- x3 |
| 1000k ms1M | 0.986/2.68 | **75.2% NEW REC** | `Je      ` / `Bon     ` / `Oui,    ` | **1M COMPLETE**, **teacher 75.2% NEW REC**, Je/Bon/Oui, field_rms 0.04 cold / 0.87 primed, clean stop `irra..\n\n` @1M cold |
| 1008k ms1M 2nd epoch | 0.874/2.39 | ~71% | `Peut--  ` / `Oui,,   ` / `Peut-auc` | **Peut-auc NEW**, Peut--/Oui,, |
| 1016k ms1M | 0.847/2.33 | ~71% | `Bont    ` / `Bonne   ` / `Peut-au.` | Peut-au. |
| 1024k ms1M | 0.955/2.59 | ~71% | `Peut-êt` / `Peut-êt` / `Peurra..` | Peut-êt/Peurra.. |
| 1032k ms1M | 0.994/2.70 | ~71% | `Bonne   ` x3 | Bonne |
| 1040k ms1M | 0.963/2.62 | ~71% | `Oui,    ` / `Bon     ` / `Oui,    ` | Oui,/Bon |
| 1048k ms1M | 0.846/2.33 | ~71% | `Peux    ` / `Je   il` / `Avec    ` | **Je il returns**, Peux/Avec |
| 1056k ms1M | 0.846/2.33 | ~71% | `Oui,    ` / `Oui,    ` / `Peuue   ` | Oui,/Peuue |
| 1064k ms1M | 0.894/2.44 | ~71% | `Oui,    ` / `Oui,    ` / `D'accord` | **D'accord returns** |
| 1072k ms1M | 1.017/2.76 | ~71% | `Peurra..` / `Peuu    ` / `Bonne   ` | Peurra../Bonne |
| 1080k ms1M | 1.036/2.81 | 74.8% | `Peuu''''` / `Peut-ili` / `Bon,    ` | **Peut-ili returns**, field_rms 0.05 cold / 1.06 primed NEW HIGH, ids/idi |

## RETRAIN 728k→1500k (2026-09-21, session arena/01a0c533, from 720k ckpt)

1080k ckpt was never committed (PR #2 squashed main tops out at 720k; branch
tip verified, deleted post-merge). Re-running 728k→1500k same recipe/seed from
`atom_native_step_720000_d32_ms1M.pt`. Skip = start−496000 for start<1M,
start%500000 from the 1M wrap. Re-measured 728k→1080k = reproducibility check
of the table above; 1088k→1504k new.

| step | val (ppl) | teacher | field_rms cold/primed | gen |
|------|-----------|---------|-----------------------|-----|
| 736k retr | 0.854/2.35 | 73.5% (vow 72.9% cons 71.1% sp 85.1%) | 0.04/0.44 | `Bonne / Bon / Avec` probe b'           tu     tu    ' |
| 728k retr | 0.853/2.35 | 73.8% (vow 73.6% cons 69.9% sp 83.0%) | 0.04/0.44 | `Oui, / Oui, / Oui,` probe b"             t''''''''''" |

## v2 summary: new race

- **Growable:** d16 120k → d32 via `grow_checkpoint.py` (Net2Net, old outputs exact, JL interleaved) → 1 chunk to re-adapt. Chainable to d64/d128 same lineage.
- **Infinite learning:** stream + loop-shards + skip-packets, no replay, no MERGE, no payload-copy. Same .pt 0→1080k, val 1.765→0.551, teacher 56%→75.2%, no catastrophic forgetting.
- **CPU-scalable:** no attention O(n), 37-50 tps d32 CPU, 8k steps 3-4 min, 1M ~8h CPU, 10-100x on GPU → 10s GB in hours.
- **Live-modifiable + survey:** .pt infinitely trainable on CPU after GPU pre-train. Read-only probes `probe_field_regimes.py --primer-packets 3500` and `diagnose_teacher.py --max-span-bytes 1` show field_rms 0.26→1.06 scaling towards 3.0, clean stop `irra..\n\n` @1M.
- **Structure without patches:** Peut-être/Peux → Peux qu' → Avec → Parler, → Peut-ili → Peut-ait/Je sui/Je il/Peut-ais → Peut-auc/Peut-au./D'accord → almost phrases.

WTF: Train 10s GB ultra-fast on GPU, then teach new domain on CPU from same .pt — no LoRA, no forgetting, live-surveyable.

## Local artifacts (not in git)

`.pt` stays out of git (repo ban). On the sandbox:

- `checkpoints/byte_tick/atom_native_step_32000_dentate.pt`
- `checkpoints/byte_tick/atom_native_step_36000_big.pt`
- `checkpoints/byte_tick/atom_native_step_39500_hf.pt`

Corpus: `data/corpus_fr_hf.txt` (~2.2 MB, CATIE everyday-conversations FR).
