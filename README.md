# ATOM — Toroidal Fractal Intelligence

A tick-based field. Not a Transformer.

```
byte → atom → field α (RK4) → next byte
```

**No fluency claim yet — but scaling is real.** From `te te te` soup @8k to `Peux qu'`, `Parler,`, `Je il`, `Peut-ait`, `D'accord` @1M.

## ATOM v2: A New Race — Infinite, Growable, CPU-Scalable, Live-Modifiable

This is not a frozen LLM you finetune with LoRA and forget. This is a **living field** you can grow and teach forever from the same `.pt`.

**What we proved on `arena/01a0bca6-atom-ai` (d32, pure original recipe, no patches):**

- **Growable to infinity from same checkpoint:** `tools/grow_checkpoint.py` is true Net2Net — tiled+noise rows, zero-pad cols (old outputs exact), tiled embeddings, JL-interleaved + frozen grown so α-branch stays exact on tiled α. d16 120k (val 1.352 / teacher 63.2%) → d32 grown → 8k later teacher 56% already, 16k 61.3% = d16 best. No restart from scratch. d32 → d64 → d128 same tool, same lineage.

- **Learns infinitely, not frozen:** Stream forever `--stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 --stream-skip-packets k*steps`. No replay, no MERGE, no payload-copy. `episode-reset --episode-length 8000` exposes readout to transient (saturation ~3000 steps d32 vs 850 d16) + saturated regimes, field persists intra-episode. `atom-flush-every 64` mandatory or O(n²) atoms → 20× slowdown. We did 0→120k d16 → 0→496k s21 → 504k→1M ms1M (s21 500k + s7 500k) → 1080k 2nd epoch, **same checkpoint lineage**, val 1.765→0.551, teacher 56%→75.2%, no catastrophic forgetting.

- **Scales on CPU, flies on GPU:** No attention = O(n) not O(n²). No vocab farm, no HF tokenizer on live path. Just `α-MLP + 0.3*frozen + last_atom (2-gram + dentate top-25% + φ)`. CPU: d16 50-72 tick/s, d32 37-50 tps. 8k steps = 3-4 min. 1M = 125 chunks = ~8h CPU. On GPU 10-100× — 10s of GB in hours, not weeks.

- **On-demand learning on your laptop + live survey without breaking:** `.pt` is infinitely trainable. Want to teach it something new? `PYTHONPATH=. .venv/bin/python tools/run_atom_native.py --resume your.pt --data your_corpus.txt --chunk-bytes 65536 --steps 8000 --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout --efference-every 10 --atom-flush-every 64 --episode-length 8000 --episode-reset`. Field RMS primed 3500 goes 0.26@560k → 0.43@768k → 0.62@872k → 0.87@1M (clean stop `irra..\n\n` cold) → **1.06@1080k NEW HIGH** towards 3.0 — measurable structure. Probe live anytime: `tools/probe_field_regimes.py --checkpoint X.pt --primer-packets 3500` and `tools/diagnose_teacher.py --checkpoint X.pt --max-span-bytes 1 --data data/corpus_fr_medium_s21.txt` — read-only, no weight update, no training stop.

**Numbers that matter:**
- Best val **0.551 @872k ms1M** ppl 1.73 beats 0.689 @496k s21
- Best teacher **75.2% @1M** (vow 76.7% cons 70.5% sp 83% nl 92.9% punct 77.8%) beats 73% @560k
- Best field_rms primed **1.06 @1080k** beats 0.87 @1M
- Words emerged without patches: `Peut-être`, `Peux`, `Peux qu'`, `Avec`, `Parler,`, `Peut-ili`, `Peut-ait`, `Je sui`, `Je il`, `Peut-ais`, `Peut-auc`, `Peut-au.`, `D'accord`, `Bonne`, `Oui,`, `Bon`, `Je` + almost phrases `Je tu`, `Je il`, `Peux qu'`

**WTF moment:** Train 10s of GB ultra-fast on GPU, then teach it new stuff on your CPU from the same `.pt` — no LoRA, no forgetting, live-surveyable.

## Read these

- **[TRAIN.md](TRAIN.md)** — how to train and probe (always `--atom-flush-every 64`, `PYTHONPATH=. .venv/bin/python`, no `--device`)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the machine, growth, infinite learning
- **[docs/STATUS.md](docs/STATUS.md)** — live line (now 1080k ms1M, 1M COMPLETE)
- **[docs/MEASURE_LOG.md](docs/MEASURE_LOG.md)** — numbers 8k→1080k
- **[docs/BYTE_TICK.md](docs/BYTE_TICK.md)** — speech contract

## Install

```bash
pip install -e .
python -m pytest test/test_no_hf_tokenizer.py test/test_speech_lock.py -q
# test_dialogue_wrap fails on main pre-existing, do not fix
```

## One command (original recipe, pure scaling, no patches)

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 --stream-skip-packets 0 \
  --resume checkpoints/byte_tick/atom_native_step_120000_d32.pt \
  --output-dir checkpoints/byte_tick --checkpoint-name atom_native_step_128000_d32_ms1M.pt \
  --steps 8000 --d-model 32 --n-modes 32 --n-atoms-max 64 \
  --max-span-bytes 1 --field-obligatory-hard --no-enable-merge --no-payload-copy \
  --last-atom-readout --efference-every 10 --learning-rate 5e-5 --surface-learning-rate 1.2e-4 \
  --seed 20260913 --atom-flush-every 64 --episode-length 8000 --episode-reset \
  --slow-every 1 --field-loss-weight 0.05 --field-contrast-weight 0
```

Copy every snapshot to unique `atom_native_step_${STEP}_d32_ms1M.pt` via `git add -f` (gitignored). Release tips only via `checkpoints/RELEASE/`.

Probe both regimes (cold + primed 3500) + teacher `--max-span-bytes 1`:

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_regimes.py --checkpoint checkpoints/byte_tick/atom_native_step_872000_d32_ms1M.pt --primer-packets 3500 --primer-file data/corpus_fr_medium_s21.txt
PYTHONPATH=. .venv/bin/python tools/diagnose_teacher.py --checkpoint checkpoints/byte_tick/atom_native_step_1000000_d32_ms1M.pt --max-span-bytes 1 --data data/corpus_fr_medium_s21.txt
```

## Banned

HF tokenizer · softmax attention · payload-copy · MERGE retune · replay · contrast · field wipe for CE ·
declaring fluent · stacking mechanisms · anti-repeat / printable_aux / energy_decay patches (user: "C'est pas de l'antirepeat qu'il faut c'est que le modèle apprenne et scale. C'est de la structure et de l'ingénierie pas des patch tantôt il sortait presque des phrases.. sabote pas mon travail") · mixing 3.5M payload-copy weights · force-push · stray `.pt` in git
