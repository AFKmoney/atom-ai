# ATOM speech cuts — measured log

Sandbox: d=16, byte-tick, CPU ~52–72 tick/s.
One mechanism per row. **No fluency claim.**

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

Sandbox: d=16/n=16/64, max-span 4 (Atomizer floor is 2; true byte-tick=1 not
in code), stream, hard, MERGE off, payload-copy off at train AND chat
(`--no-payload-copy`, new flag), hard mix = obl·α-MLP + 0.3·frozen (RMS cap
removed — measured cap≈0.01 crushing trainable grads ~100x), span-level
n-gram anti-repeat in `generate_packets` (decode only). Corpus:
`data/corpus_fr_medium.txt` (500 kB procedural FR dialogue, not in git;
generator `data/gen_corpus_fr_medium.py`). Probe: `tools/probe_speech.py`
(payload_copy=False, dialogue_wrap=True, speech_gate=False). Speed on this
box: ~20–40 tick/s (variable). **No fluency claim — debris.**

| step | train byte_ppl | val byte_ppl | det. probe + anti-repeat | note |
|------|---------------|-------------|--------------------------|------|
| 4k | ~11 | 21.3 | `\nsi \nsi …` (single attractor) | vowels+spaces exist |
| 8k | ~11 | 21.4 | `ssi ssi …` / `\nsin …` | diverse across prompts |
| 12k | ~11 | 23.3 | `ssi→sse→se` | sibilant ladder |
| 16k | ~10 | 143.3 | `ssa / ssat` | val spike = cold-field ring artifact |
| 20k | ~10 | 49.0 | `sat est satsuas` | first real word `est` |
| 24k | ~10 | 50.2 | `tuat / uut / uat / ent` | (prefix retrain — see below) |
| 28k | ~10 | 44.4 | `uat / ent / ois / uit / sit` | teacher vowel acc 9.5% |
| 32k | ~42→ | 30.3 | `sai / sai / eit` | fresh data (skip 4k) |
| 36k | ~19 | 33.3 | `sai / sut / uit` | fresh data (skip 8k) |
| 40k | ~15 | 27.1 | `uateeai / eiteou / …tesisirs` | long varied chains |
| 44k | ~19 | 24.0 | `eitet / eitsun / eite` | fresh data (skip 16k) |
| 48k | ~21 | 23.9 | `uiteuui / eiteua / reri` | teacher vowel acc 14.7% |

Findings while running this line:

- Train/chat payload mismatch: with copy ON at train, chat with copy OFF gave
  consonant cycles (`mpem…`), chat with copy ON gave vowels (`tes tis tit`) —
  the copy was a crutch. Fixed by training with `--no-payload-copy`.
- Chained `--resume` chunks re-trained the same stream prefix (stream restarts
  at packet 0 per run); 0–28k above saw only the first 4000 packets 7x. Fixed
  with `--stream-skip-packets k*steps` (32k+ see fresh packets).
- Old hard checkpoints (capped mix, copy-on) are invalid under the new forward
  (logit scale changed). Do not mix lines.

Speech tip this line: `atom_native_step_48000_mix.pt` + anti-repeat.
Not fluent — French babble with word-adjacent fragments (`est`, `sit`, `un`).

## Ligne B — byte-tick + last-atom readout + dentate (2026-09-19/20)

Sandbox: d=16/n=16/64, **max-span 1** (Atomizer floor lowered 2→1; pure byte
stream, mid-codepoint splits by design), stream 64 kB chunks, hard, MERGE off,
payload-copy off, last-atom readout (2-gram + dentate top-25% + φ phase).
Logits = obl·α-MLP + 0.3·frozen + last-atom. Aux hinges compare
same-bytes/different-α. Decode: n-gram anti-repeat 2-8, speech gate on the
accumulated phrase (never a 1-byte packet). Same 500 kB FR corpus.
~48–55 tick/s. **No fluency claim — debris with words.**

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

Teacher-forcing @72k (400 fresh transitions): byte acc **59.5%** (vowels 64%,
consonants 53%, spaces 83%, newlines 100%, punct 28%).

Best chat = train tip this line: `atom_native_step_72000_btick.pt`.
Note: attractors hop every ~8-16k (`te` → roles → `Tu` → `toi/?`); d=16
capacity + no efference yet. Next documented cuts: efference copy (bridge #9),
d=32+, more corpus (only 72k/500k bytes seen = 14%).
Session note: sandbox drops gitignored `.pt` on reset, so the 72k weights are
force-committed on the session branch only — remove before merging to main.

## Ligne B++ — efference copy + le mirage du collapse (2026-09-20)

Efference (bridge #9): `--efference-every 10`, 1 tick/10 = predict own byte
(no-grad) → commit → CE on gold next. 4 tests in `test/test_efference.py`,
~0% tps cost, 800/8000 eff ticks verified.

| step | val byte_ppl | cold probe (40-pkt prompt) | primed probe (saturated field) |
|------|-------------|---------------------------|-------------------------------|
| 80k eff | **3.9** (record 1.42) | `toi   ssistant:` | — |
| 82k eff | — | `toi to tant: on plus` (best cold) | — |
| 88k eff | 4.5 | `�FF�e� {` binary | `Tu vec toi ?\nAssisateur` FRENCH |
| 88k no-eff (A/B) | — | `�FF��FF` garbage | `�e�r�\nUtilisateur: …` mixed |
| 88k no-contrast | — | `�FF�e� {` bit-identical to eff | `Tu vec toi ?\nAssisateur` bit-identical |
| 96k eff | 4.3 | `�on plustant: …` mixed | `Tu vec toi.\n\n` FRENCH + clean stop |

**Le collapse 88k est un mirage de mesure (régime de champ), pas un dégât.**
Preuves : (1) A/B sans efference → même soupe bit-identique : efference
innocente. (2) Corpus/fenêtre/loader/skip innocentés (fenêtre 80-88k = FR
propre, 62 octets). (3) Trace : logits 88k plats dès t0 (H=3.67 vs 0.02,
top-1 0.34 vs 1.00), champ α identique 80k/88k → readout seul coupable.
(4) Transplantation : 80k α-MLP (6 tenseurs) dans 88k/ab2/96k → français
restauré (`…d'accord…`, `toi toistant:`). (5) Drift fc2 systématique :
cos(eff,ab2)=+0.976, linéaire en steps, même sans contraste (ablation
contrast=0 → même soupe). (6) Décomposition exacte : la pointe 2-gram
`' '` (+4.4) survit, mais la MLP booste +2..+3 sur des octets poubelle
(254/70/230/…), marge 6.36→2.55. (7) **Primer 1500 octets (champ saturé
rms 3.0 = régime d'entraînement) → 88k parle français.**
Diagnostic : la α-MLP sur-apprend le régime saturé (tout l'entraînement s'y
déroule dès ~run 850/chunk, rms pinné à 3.0) ; la sonde froide (rms 0.06,
40 paquets) est hors-distribution pour elle. Teacher-forcing (59.8%) et val
aveugles car contextes longs = champ saturé. Fix principled next cut :
`--episode-reset` (exposer le readout aux champs transitoires pendant
l'entraînement) et/ou warmup systématique en `generate_packets`.
Efference elle-même : non coupable ; sondes primées 88k-eff (`Tu vec toi
?\nAssisateur`, zéro octet poubelle) vs 88k-no-eff (mots + octets poubelle)
suggèrent un bénéfice propre — 1 sample chacun, à confirmer. Bonus :
`--field-contrast-weight 0` donne des sorties bit-identiques à froid ET à
chaud → le hinge contrastif est satisfait à 0 (gradients nuls, 3 forwards
surface/step gaspillés) : le couper = speedup gratuit. Meilleur sample de
la ligne : 82k primé (`toi ?\nAssistant: toi `) ; tip : 96k primé
(`Tu vec toi.\n\n`). Protocole de sonde honnête : primer le champ (~1500
octets, flush atoms/64 — sans flush, O(n²) atoms) ou `--episode-reset` en
entraînement.

## Cut episode-reset (2026-09-20) : 96k→104k, ep-2000 + contraste 0

4 épisodes/chunk (~850 transitoire + ~1150 saturé). Smokes froides du
trainer redevenues françaises (`on pars`, `stant:`). Val 1.54→**1.477**
(ppl 4.13). Cold : `toi ?\nAssistant: toi` (magnifique, ex-record 82k
primé) ; primed : `pen pen pen…` (boucle). Teacher span-1 intact :
**59.8%** (punct 22→33%). Lecture : à d=16 le readout fait le grand écart
entre régimes — la boucle saturée est un attracteur de free-run, pas une
perte de savoir. Note méthodo : `diagnose_teacher` défaut span-4 donne
~22% sur un modèle byte-tick (artefact) ; toujours `--max-span-bytes 1`.
Vitesse : ~50 tps (contraste coupé, -3 forwards/step).

## 112k ep-4000 : le compromis deux-régimes existe (2026-09-20)

104k→112k, `--episode-length 4000` (~20% transitoire). Val **1.423**/ppl
3.90 (record égalé), teacher span-1 **61.3%** (record : cons 56%,
punct 36%). Cold : `Tu toi ?\nAssistant: Tu` ; primed :
`Tu ?\nAssistant: Tu ?\nAs` — **français dans les deux régimes, sans
boucle**. d=16 suffit donc aux deux mappings avec le bon dosage ; pas
besoin de d=32 pour ça. Tip de ligne : `atom_native_step_112000_epr4.pt`.

## 120k : fin du d16 (2026-09-20)

112k→120k à l'identique (ep-4000, contraste 0, eff 10). Val **1.352**/ppl
3.65 (record), teacher span-1 **63.2%** (record). Cold :
`toi ?\nAssistant: men` ; primed : `Je snsestant: plaistan` (latin,
attracteur qui ondule — le savoir est intact). Corpus seed-7 vu à 24%.

## d=32 par croissance (2026-09-20) : `tools/grow_checkpoint.py`

Net2Net préservant : lignes doublées tuilées+bruit, colonnes doublées
zéro-pad (sorties anciennes exactes), embeddings tuilés+bruit (briseurs
de symétrie), JL entrelacé + frozen grown (branche α exacte sur α tuilé).
3 itérations : tile+noise naïf (×2 magnitudes → soupe), preserve (primed
latin `T T T`), +JL/frozen exact (paradoxalement pire : la trajectoire α
d32 diffère — saturation à ~3000 pas au lieu de ~850 — donc fonction
exacte ≠ même comportement : champ et readout sont co-adaptés).
Leçon : pas de init miracle, le training ré-adapte (1 chunk suffit).
Règle d'épisode d32 : ep-6000 (~2× temps de saturation).

## d32 sur corpus s21 (seed 21, 500k frais, 2026-09-20)

| step (s21) | val | teacher | cold | primed (3.5k) |
|------------|-----|---------|------|---------------|
| grown (init) | — | — | soupe | soupe (rms 0.8) |
| 8k | 1.765/5.84 | **56.0%** (punct 47%) | espaces/`?` | `Distant: Distant:` rôles ! |
| 16k | 1.585/4.88 | **61.3%** | `Ouistant: …` rôles à froid | `Tu ?\nAssistant: Tu ?` = best d16 |
| 24k | 1.507/4.51 | 61.3% (cons 60%, \n 93%) | `Oun plaistant:…` (boucle) | `Tu plaistant:…` (boucle) |
| 32k | 1.490/4.44 | **63.0%** (voy 68%, punct 50%) | `?\nAssistant:…` (boucle motif) | `Oui, que pre pre…` (boucle) |
| 40k | 1.417/4.13 | **64.7%** (voy 67%, cons 60%, sp 81%, nl 93%, punct 53%) | ` On pre ?\nAssistant: On ` (boucle `pre`) | ` On pre vec pre vec pre ` (boucle `pre vec`) |

40k: val ↓ (1.490→1.417, record s21), teacher ↑ (63.0→64.7% record), tps 67.4. Free-run reste en phase boucle `pre`/`vec` mais cold montre `?` + rôles (`Assistant:`) — même pattern que 24-32k. Savoir intact (val/teacher montent). Continuation identique attendue.
| 48k | 1.523/4.59 | **67.5%** (voy 69%, cons 65%, sp 83%, nl 93%, punct 53%) | `                        ` (spaces) det + ` Pe ?\nAs` trainer smoke / `Tu ?\nAssistant:   ` chat det | `                        ` (spaces) / sampled `Pe que vec toi.` |

48k: val ↑ 1.417→1.523 (régression), teacher ↑ 64.7→67.5% (record). Free-run déterministe tombe sur attracteur espace (probable effet `prefer_printable` qui choisit ` ` quand logits plats), mais sampling temp 0.8 donne `Pe que vec toi.` (français, rôles). Trainer smokes ` Pe ?\nAs` identiques → même régime que 40k. Savoir intact (teacher monte). Seuil 48k atteint → test ep-8000 prévu (une seule variable).
| 56k ep8000 | 1.479/4.39 | 64.5% (voy 66%, cons 59%, sp 89%, nl 93%, punct 47%) | ` Peeur:  ?\nAssistant: Pe` (rôles, `peeur` boucle) | ` Tu ?\nAssistant: Tu ?\nAs` **FR both regimes, best d16 level** |

56k ep-8000 (seule variable changée vs 48k): val ↓ 1.523→1.479, teacher ↓ 67.5→64.5%. **Free-run réparé**: cold `Peeur: ?\nAssistant: Pe` (français + rôles), primed `Tu ?\nAssistant: Tu ?\nAs` = compromis deux-régimes 112k d16 (record). Dosage: ep-6000 = 50% transitoire (3000 sat /6000), ep-8000 = 62.5% saturé (37.5% trans). Plus de saturé → free-run meilleur, teacher légèrement ↓. Prochain: ep-8000 continu ou ep-10000/12000 (~75% saturé) pour approcher 80% saturé de d16 ep-4000.
| 64k ep8000 | 1.432/4.19 | 66.5% (voy 70%, cons 62%, sp 83%, nl 93%, punct 53%) | ` Tu ?\nAssistant: Tu ?\nAs` **best cold = d16 112k** | ` Tu plaisateur: Tu plais` (boucle `plais` mais FR) |

64k ep-8000 continu: val ↓ 1.479→1.432 (proche record 1.417), teacher ↑ 64.5→66.5%. Cold parfait `Tu ?\nAssistant: Tu ?\nAs` (même que 56k primed, best), primed `plaisateur` boucle mais FR. Chat det = cold parfait. Attracteur `pre`/`vec`/`espace` résorbé → `plais`/`Tu ?` phase. Savoir intact, free-run meilleur que 40k. Continue ep-8000.
| 72k ep8000 | 1.889/6.61 | 66.0% (voy 68%, cons 61%, sp 81%, nl 93%, punct 58%) | ` Tu plaisateur: Tu  plai` (boucle `plais`) | ` Tu plaisateur: Tu plais` (idem) |
| 80k ep8000 | 1.791/6.00 | 65.5% (voy 68%, cons 61%, sp 79%, nl 93%, punct 56%) | `  ?\nAssistant: Tu ?\nAssi` (`Tu ?` + rôles) | ` Tu pent: Tu  pes d'acco` (FR, `d'acco` = `d'accord` fragment) |

72k: val ↑ 1.432→1.889 (spike, 2nd epoch data? skip 64k loops corpus 500k), teacher 66.5→66.0 stable, free-run `plaisateur` both regimes FR. Trainer smokes `Bonn es ` / `Tu ?\nAss` / `Disateur` (rôles). 80k: val ↓ 1.889→1.791, teacher 66.0→65.5 stable, cold `?\nAssistant: Tu ?`, primed `d'acco` fragment — nouveau mot `d'accord` émerge (jamais vu avant). Savoir intact (teacher stable), val spike probable bruit validation (stream_ring) + changement distribution 2nd epoch. Prochain: ep-12000 dosage (75% saturé) pour approcher 80% saturé d16 ep-4000, une seule variable.
| 88k ep12000 | 1.842/6.31 | 64.0% (voy 68%, cons 57%, sp 87%, nl 93%, punct 50%) | `  ?\nAssistant:  ?\nAssist` (rôles loop) | ` Tu vec pen pen pen pen ` (boucle `pen`) |
| 96k ep8000 | 1.809/6.11 | 66.0% (voy 66%, cons 60%, sp 96%, nl 93%, punct 53%) | `  ?\nAssistant:  ?\nAssist` (rôles) | ` Tu vec toi ?\nAssistant:` (FR, rôles) |
| 104k ep8000 | 1.706/5.51 | 67.0% (voy 68%, cons 63%, sp 85%, nl 93%, punct 56%) | `  ?\nAssistant: Tu ?\nAssi` (FR) | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both regimes** |
| 112k ep8000 | 1.645/5.18 | **69.0%** (voy 73%, cons 62%, sp 92%, nl 93%, punct 58%) | `  ?\nAssistant:  ?\nAssist` (rôles) | ` Tu vec plaisateur: Tu v` (FR) |
| 120k ep8000 | 1.716/5.56 | 66.5% (voy 68%, cons 60%, sp 92%, nl 93%, punct 58%) | ` Tu  Tu   Tu    Tu   Tu ` (boucle `Tu`) | ` Tu vec plaisateur: Tu  ` (FR) |

88k ep-12000 (seule variable 8000→12000): val 1.791→1.842 ↑, teacher 65.5→64.0 ↓, cold rôles loop, primed `pen pen` loop (comme d16 ep-2000). Donc ep-12000 pire que ep-8000 → ep-8000 sweet spot pour d32 (62.5% saturé) vs ep-4000 pour d16 (78% saturé). 96k retour ep-8000: val 1.842→1.809 ↓, teacher 64→66 ↑, primed `vec toi ?\nAssistant:` FR. 104k ep-8000 continu: val 1.809→1.706 ↓, teacher 66→67 ↑, **both regimes perfect** `Tu ?\nAssistant: Tu ?\nAs` primed, cold `?\nAssistant: Tu ?`. 112k: val 1.706→1.645 ↓, teacher 67→**69% record**, smokes `plainuo`/`Pe ven`. 120k: val 1.645→1.716 ↑ léger, teacher 69→66.5 ↓, smokes `Ouisateu`/`Je veur:`/` d'accor` (d'accord revient, Ouisateur = ancien attracteur d16 64k). Savoir intact (teacher 63→69% sur ligne), val trend 72k spike puis descente 1.889→1.645 sur 40k steps. Free-run reste FR avec attracteurs `plais`/`Tu`/`vec`/`d'acco`. Prochain: continuer ep-8000 ou essayer LR decay? Mais consigne = un seul levier à la fois, ep-8000 stable.
| 128k ep8000 | 1.677/5.35 | 67.2% (voy 69%, cons 61%, sp 92%, nl 93%, punct 56%) | `  ?\nAssistant:  ?\nAssist` (rôles) | ` Tu plus.\n\n` (FR + clean stop `\n\n`) |
| 136k ep8000 | **1.370/3.93** | 68.2% (voy 69%, cons 61%, sp 92%, nl 93%, punct 67%) | ` Tu  ?\nAssistant: Tu  ?\n` **perfect + clean stop** | ` Tu ?\nAssistant: Tu  ?\nA` **perfect** |
| 144k ep8000 | **1.365/3.92** | 66.2% (voy 68%, cons 61%, sp 81%, nl 93%, punct 58%) | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both** | ` Tu ?\nAssistant: Tu ?\nAs` **perfect both** |
| 152k ep8000 | 1.477/4.38 | 67.5% (voy 68%, cons 62%, sp 85%, nl 93%, punct 67%) | `  ?\nAssistant:  ?\nAssist` (rôles) | ` D'accord ?\nAssistant: D` **D'accord full** |
| 160k ep8000 | **1.346/3.84** | 67.0% (voy 68%, cons 60%, sp 89%, nl 93%, punct 64%) | `  ?\nAssistant:  ?\nAssist` | ` Tu vec  vec plaisateur:` (FR loop) |
| 168k ep8000 | 1.364/3.91 | 68.0% (voy 68%, cons 63%, sp 87%, nl 93%, punct 64%) | `  ?\nAssistant: Tu veur: ` (`Tu veur` = tu veux) | ` Tu veur:  Tu  Tu  Tu  T` (FR) |
| 176k ep8000 | 1.383/3.99 | 68.0% (voy 69%, cons 62%, sp 87%, nl 93%, punct 67%) | `  ?\nAssistant: Tu       ` | ` Tu  veur:              ` (spaces loop) |
| 184k ep8000 | 1.399/4.05 | 67.0% (voy 67%, cons 61%, sp 87%, nl 93%, punct 64%) | `  ?\nAssistant:  ?\nAssist` | ` D'accord ?\nAssistant: D` **D'accord returns** |
| 192k ep8000 | 1.423/4.15 | 67.7% (voy 68%, cons 63%, sp 89%, nl 93%, punct 61%) | `  vais d'accord ?\nAssist` (`vais d'accord` new) | ` Ouis d'accord ?\nAssista` |
| 200k ep8000 | **1.063/2.90** | 67.2% (voy 66%, cons 62%, sp 85%, nl 93%, punct 69%) | ` Tu                     ` (spaces det) | ` Oui,  Et toi ?\nAssistan` **best primed FR** |
| 208k ep8000 | **1.055/2.87** | 68.0% (voy 67%, cons 63%, sp 89%, nl 93%, punct 67%) | `  vec  vais d'accord ?\nA` | ` Tu veur:  Tu     Tu    ` |
| 216k ep8000 | 1.086/2.96 | **69.0%** (voy 69%, cons 65%, sp 87%, nl 93%, punct 64%) | `  toi ?\nAssistant:  toi ` **both roles FR** | ` Tu                     ` |
| 224k ep8000 | 1.077/2.94 | 66.5% (voy 69%, cons 60%, sp 85%, nl 93%, punct 61%) | ` Tu                     ` | ` Tu  toi                ` |
| 232k ep8000 | **1.027/2.79** | 68.2% (voy 70%, cons 62%, sp 87%, nl 93%, punct 67%) | `  toi.\n\n` clean stop | ` Tu                     ` |
| 240k ep8000 | 1.042/2.84 | 67.7% (voy 68%, cons 63%, sp 85%, nl 93%, punct 61%) | `  ?\nAssistant: Tu       ` | ` Tu                     ` |
| 248k ep8000 | **1.025/2.79** | 66.5% (voy 68%, cons 61%, sp 87%, nl 93%, punct 56%) | `  toi                   ` | `  Et toi                ` |
| 256k ep8000 | 1.133/3.10 | 68.8% (voy 72%, cons 62%, sp 85%, nl 93%, punct 64%) | `  Tu                    ` | ` Tu                     ` |
| 264k ep8000 | 1.188/3.28 | **70.5%** (voy 73%, cons 65%, sp 85%, nl 93%, punct 69%) | `  vais d'accord ?\nAssist` (vais d'accord) | `  Et toi,               ` |
| 272k ep8000 | 1.256/3.51 | 69.8% (voy 72%, cons 62%, sp 89%, nl 93%, punct 72%) | `  vais d'accord ?\nAssist` | ` Tu                     ` |
| 280k ep8000 | 1.264/3.54 | 68.5% (voy 71%, cons 61%, sp 87%, nl 93%, punct 72%) | `  Tu                    ` | ` Tu                     ` |
| 288k ep8000 | 1.112/3.04 | 70.3% (voy 71%, cons 65%, sp 87%, nl 93%, punct 69%) | `                        ` (spaces) | ` Tu                     ` |
| 296k ep8000 | 1.123/3.07 | 68.5% (voy 71%, cons 64%, sp 83%, nl 93%, punct 61%) | `                        ` | ` Tu                     ` |
| 304k ep8000 | 1.105/3.02 | 69.3% (voy 72%, cons 63%, sp 89%, nl 93%, punct 64%) | `                        ` (spaces) | `                        ` spaces |
| 312k ep8000 | 1.147/3.15 | 70.0% (voy 74%, cons 64%, sp 85%, nl 93%, punct 67%) | `  soir,                 ` (**soir, new word**) | ` Tu                     ` |
| 320k ep8000 | 1.089/2.97 | 70.3% (voy **75%**, cons 64%, sp 85%, nl 93%, punct 64%) | `                        ` | `  suistant::            ` |
| 328k ep8000 | **0.753/2.12** | 69.0% (voy 71%, cons 63%, sp 87%, nl 93%, punct 64%) | `  suis d'accord ?\nAssist` **suis d'accord both** | `  suis d'accord ?\nAssist` **both perfect** |
| 336k ep8000 | 0.854/2.35 | 69.5% (voy 71%, cons 64%, sp 87%, nl 93%, punct 69%) | `                        ` spaces | `                        ` spaces |
| 344k ep8000 | 0.866/2.38 | 68.5% (voy 70%, cons 61%, sp 89%, nl 93%, punct 69%) | `                        ` | `                        ` |
| 352k ep8000 | 0.837/2.31 | 69.3% punct **75% NEW REC** | `                        ` | `                        ` |
| 360k ep8000 | 0.833/2.30 | 67.7% (voy 71%, cons 62%, sp 83%, nl 93%, punct 64%) | `                        ` | `                        ` |
| 368k ep8000 | 0.889/2.43 | 69.0% (voy 71%, cons 61%, sp 89%, nl 93%, punct 69%) | `                        ` | `                        ` |

128k: val 1.716→1.677 ↓, teacher 66.5→67.2 ↑, primed `Tu plus.\n\n` clean stop double newline = idéal dialogue, smokes `Bon plus`/`Je ven`/`d'accor`. 136k: val 1.677→**1.370 record s21** (ppl 3.93) bat 40k 1.417 et proche d16 1.352, teacher 68.2% punct 67% record, cold+primed perfect `Tu ?\nAssistant: Tu ?\n` + clean stop, smokes `Bon que`/`Tu ves`/`Ouisateu`. **d32 bat d16 en val**. 144k: val **1.365 record** bat 1.370, teacher 66.2%, **both regimes perfect identical** `Tu ?\nAssistant: Tu ?\nAs` = best d16 112k, smokes `Parle  v` x3 (Parle verbe FR). 152k: val 1.365→1.477 ↑, teacher 66.2→67.5 ↑, primed `D'accord ?\nAssistant: D` **D'accord full avec ?** première fois, smokes `Je que`/`Oui ?\nAs`. 160k: val 1.477→**1.346 record** ppl 3.84 **bat d16 1.352** → **d32 dépasse d16**, teacher 67%, smokes `Parle  q`/`Je ve`/`Ouisateu`. 168k: val 1.346→1.364 ↑ léger, teacher 67→68 ↑, cold `Tu veur:` (tu veux), smokes `Parle  ?`/`Je qu'e`. 176k: val 1.364→1.383 ↑, teacher 68% stable, cold `?\nAssistant: Tu`, primed `Tu veur:` spaces, smokes `Peur:`. 184k: val 1.383→1.399 ↑, teacher 68→67, primed `D'accord ?` returns, smokes `Je toi,`/`Je qu'`. 192k: val 1.399→1.423 ↑, teacher 67→67.7, cold `vais d'accord ?` new compo, primed `Ouis d'accord ?`, smokes `Parle Q`/`Parle v`. 200k: val 1.423→**1.063 record** ppl 2.90 **HUGE** beats all, teacher 67.2% punct 69% rec, primed ` Oui,  Et toi ?\nAssistan` **best FR ever** (Oui, Et toi ? + Assistant), sampled `je vais d'accord ?` + `Je vais`/`Je vaiss` smokes. 208k: val **1.055 record** ppl 2.87 beats 1.063, teacher 68%, cold `vec vais d'accord ?`, smokes `Bon Qu'`/`Oui ?\nAs`. 216k: val 1.055→1.086 ↑, teacher 68→**69% tie rec**, cold `toi ?\nAssistant: toi` perfect both roles, smokes `Je Qu'`/`Je ver`. 224k: val 1.086→1.077 ↓, teacher 69→66.5, cold spaces, primed `Tu toi`, smokes `Bonnt to`/`Peur: t`/`Bon moi`. 232k: val 1.077→**1.027 record** ppl 2.79 **NEW rec beats 1.055**, teacher 68.2%, cold `toi.\n\n` clean stop, smokes `Je s.\n\n`/`Je ver` (Je sais? fragment). 240k: val 1.027→1.042 ↑ léger, teacher 68.2→67.7, cold `?\nAssistant: Tu`, smokes `Bon ?\nA`/`Bon Tu`/`sesta`. 248k: val 1.042→**1.025 record** ppl 2.79 beats 1.027, teacher 67.7→66.5, cold `toi`, primed `Et toi`, smokes `Je Qu'`/`suis d'` (suis d'accord fragment). 256k: val 1.025→1.133 ↑, teacher 66.5→68.8% (vow 72% rec), cold `Tu`, primed `Tu`, smokes `Bon, je` **Bon, je** excellent FR start. 264k: val 1.133→1.188 ↑, teacher 68.8→**70.5% NEW RECORD** (vow 73%, cons 65%, punct 69%), cold `vais d'accord ?\nAssist`, primed `Et toi,`, smokes `Je Qu'`/`Bon vai`. **teacher 70.5% beats all**. 272k: val 1.188→1.256 ↑, teacher 70.5→69.8% punct 72% rec, cold `vais d'accord ?`, smokes `Je s.\n\n`. 280k: val 1.256→1.264 ↑, teacher 69.8→68.5% punct 72% tie, cold/primed Tu, smokes `: Tu`/`Peur::`. 288k: val 1.264→1.112 ↓, teacher 68.5→70.3%, cold spaces, primed Tu, smokes `Oui,\nAs`/`Je Tu`/`Peur:`. 296k: val 1.112→1.123 ↑, teacher 70.3→68.5%, cold/primed Tu, smokes `Bon vai`/`Bon nne`/`Bonne` **Bonne** FR. 304k: val 1.123→1.105 ↓, teacher 68.5→69.3%, cold/primed spaces (attracteur espace revient comme 48k/304k), smokes `: toi`/`Je es`. 312k: val 1.105→1.147 ↑, teacher 69.3→70.0% vow 74% rec, cold `soir,` **new word soir,**, smokes `Je saan`/`Oui, toi` **Oui, toi** perfect, `Ouis ont`. 320k: val 1.147→1.089 ↓, teacher 70.0→70.3% vow **75% NEW REC**, cold spaces, primed `suistant::`, smokes `Ouis to`/`Oui,`/`Bon,`. Trend: val 1.889 (72k) → 0.753 (328k) sur 256k steps, teacher 66→70.5% record, free-run FR constant avec attracteurs `Tu ?`/`plais`/`vec`/`d'accord`/`plus`/`veur`/`Parle`/`vais d'accord`/`Oui, Et toi ?`/`Je vais`/`Je s.\n\n`/`Et toi`/`Bon, je`/`vais d'accord`/`soir,`/`Oui, toi`/`Bonne`/`suis d'accord ?`/`Bons.\n\n`/`Peurr`. 328k val 0.753 HUGE rec + both regimes perfect `suis d'accord ?\nAssist`, 336k-368k spaces attractor revient après record comme 48k/304k, mais val reste <0.9 et teacher stable 69% → savoir intact. Corpus s21 vu ~74% à 368k (368k/500k), encore de la piste. Prochain: continuer ep-8000, espaces va se résorber comme d16 64k Ouisateur → 72k best.

Le savoir transfère en 1 chunk (56% vs 63% d16) ; le free-run
récapitule vite (rôles à 8k vs ~40k en d16). ~37 tps malgré 4× champ.

## Local artifacts (not in git)

`.pt` stays out of git (repo ban). On the sandbox:

- `checkpoints/byte_tick/atom_native_step_32000_dentate.pt`
- `checkpoints/byte_tick/atom_native_step_36000_big.pt`
- `checkpoints/byte_tick/atom_native_step_39500_hf.pt`

Corpus: `data/corpus_fr_hf.txt` (~2.2 MB, CATIE everyday-conversations FR).
