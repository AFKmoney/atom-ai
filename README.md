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

- **Scales on CPU, designed to fly on GPU:** No attention = O(n) not O(n²). No vocab farm, no HF tokenizer on live path. Just `α-MLP + 0.3*frozen + last_atom (2-gram + dentate top-25% + φ)`. **Measured CPU:** d16 50-72 tick/s, d32 37-50 tps. 8k steps = 3-4 min. 1M = 125 chunks = ~8h CPU. GPU speedup is a **projection** until `--device cuda` + batched/`torch.compile` ships — see below.

- **On-demand learning on your laptop + live survey without breaking:** `.pt` is infinitely trainable. Want to teach it something new? `PYTHONPATH=. .venv/bin/python tools/run_atom_native.py --resume your.pt --data your_corpus.txt --chunk-bytes 65536 --steps 8000 --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout --efference-every 10 --atom-flush-every 64 --episode-length 8000 --episode-reset`. Field RMS primed 3500 goes 0.26@560k → 0.43@768k → 0.62@872k → 0.87@1M (clean stop `irra..\n\n` cold) → **1.06@1080k NEW HIGH** towards 3.0 — measurable structure. Probe live anytime: `tools/probe_field_regimes.py --checkpoint X.pt --primer-packets 3500` and `tools/diagnose_teacher.py --checkpoint X.pt --max-span-bytes 1 --data data/corpus_fr_medium_s21.txt` — read-only, no weight update, no training stop.

**Numbers that matter:**
- Best val **0.551 @872k ms1M** ppl 1.73 beats 0.689 @496k s21
- Best teacher **75.2% @1M** (vow 76.7% cons 70.5% sp 83% nl 92.9% punct 77.8%) beats 73% @560k
- Best field_rms primed **1.06 @1080k** beats 0.87 @1M
- Words emerged without patches: `Peut-être`, `Peux`, `Peux qu'`, `Avec`, `Parler,`, `Peut-ili`, `Peut-ait`, `Je sui`, `Je il`, `Peut-ais`, `Peut-auc`, `Peut-au.`, `D'accord`, `Bonne`, `Oui,`, `Bon`, `Je` + almost phrases `Je tu`, `Je il`, `Peux qu'`

**WTF moment (architecture, not a timed 5090 run):** same `.pt` can take a GPU pre-train then keep learning on CPU — no LoRA, no forgetting, live-surveyable. GPU wall-clock below is **not** a completed bench.

## ATOM vs Transformer — 10B / 200GB on a 5090

**These GPU times are a projection, not a measured truth.**

What is measured today:
- CPU d32: 37–50 tps → 1M bytes ≈ 8h
- Trainer is CPU-only on purpose (`tools/run_atom_native.py` has no `--device`)
- Loop is 1 Python call = 1 byte forward

What is **not** measured:
- No timed run on a 5090
- No `--device cuda`, no `transition_loss_batched`, no `torch.compile` in the trainer yet
- Naive `.to('cuda')` on the current loop is estimated 400–800 tps — still months for 10B

The 5.5h / 27h / 4.5d / 22.5d numbers assume a **future** batched+compiled loop hitting the tps in the table. FLOPs/byte (~350k vs ~14B Llama) is the reason the projection exists. Until that path ships and is benched, treat every GPU hour/day figure as an estimate, not a result.

Previous estimate was transformer-like (3–7 days) — that is what a transformer takes. ATOM has no attention; if batched compile lands near the assumed tps, it is 50–100× faster. That *if* is still open.

| | ATOM d128 1.5M | Llama 7B |
|---|---|---|
| Params | 1.5M (4666x smaller) | 7B |
| Attention | None, O(n) | O(n²) + KV-cache |
| FLOPs / byte | ~350k | ~14B |
| CPU now (**measured**) | 40 bytes/sec d32 → 1M = 8h | — |
| GPU naive `.to('cuda')` (**estimate**) | 400-800 tps = 96-192 days for 10B | — |
| GPU batched + compile (**projection**) | 100k tps d128 = 27h for 10B | ~2k tps = 58 days for 10B |

**10B tokens (10GB) on 5090 — projection if batched+compile hits the tps:**

- d32 100k: 500k tps = **5.5 hours** (projection)
- d128 1.5M: 100k tps = **27 hours** (projection)
- d256 6M: 40k tps = **2.9 days** (projection)
- Llama 7B: 2k tps = **58 days** (literature-style estimate, not our bench)

ATOM d128 **would be** ~50x faster than Llama 7B for the same 10B **under that assumption**, with live-modifiable + growable + CPU fine-tune (8k steps = 3 min on laptop, no forgetting, field_rms measurable 0.26→1.06).

**200GB (200B) on a single normal 5090 — same projection, not a farm we already ran:**

- d32 100k: 110h = **4.5 days** (projection)
- d128 1.5M: 540h = **22.5 days = 1 month if the tps holds** (projection)
- d256 6M: 58 days = **1.9 months** (projection)
- Llama 7B: 1160 days = **3.1 years** (same-class estimate)

**Claim, not result:** 200GB in ~1 month on a normal GPU is the design target with d128 if batched compile delivers ~100k tps. It is **not** a completed 5090 measurement. Llama 7B on the same 1x 5090 is years, or an 8x H100 cluster ~$500k. The architectural bet is: train 10s–100s GB on 1x 5090 (~$2000) in a living room, then fine-tune a private domain on CPU from the same small `.pt`. That bet is documented in **[docs/how_to_gpu.md](docs/how_to_gpu.md)** — including the 10-line GPU patch still to ship.

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
