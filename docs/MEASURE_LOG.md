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

## Local artifacts (not in git)

`.pt` stays out of git (repo ban). On the sandbox:

- `checkpoints/byte_tick/atom_native_step_32000_dentate.pt`
- `checkpoints/byte_tick/atom_native_step_36000_big.pt`
- `checkpoints/byte_tick/atom_native_step_39500_hf.pt`

Corpus: `data/corpus_fr_hf.txt` (~2.2 MB, CATIE everyday-conversations FR).
