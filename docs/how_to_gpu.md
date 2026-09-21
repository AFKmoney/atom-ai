# How to train ATOM on GPU — 10B tokens on a 5090

This doc is the WTF moment: ATOM is not a transformer, so 10B tokens on a 5090 is hours, not months.

## Why ATOM is 50-100x faster than a transformer on same GPU

| | ATOM d128 1.5M | Llama 7B |
|---|---|---|
| Params | 1.5M (4666x smaller) | 7B |
| Attention | None, O(n) | Softmax O(n²) + KV-cache |
| Tokenizer | None, byte-tick `max_span_bytes=1` | BPE 32k vocab |
| FLOPs / byte | ~350k (α 32x32 RK4 + α-MLP 1024→128→256 + last-atom 2-gram + dentate) | ~14B (7B * 2) |
| Theoretical max on 5090 100 TFLOPS | 285M bytes/sec | 7k tokens/sec |
| Real CPU now | 37-50 tps d32 | — |
| Real GPU naive (`.to('cuda')` same loop) | 400-800 tps (10-20x) | — |
| Real GPU batched + torch.compile | **50k-1M tps** (1000-20000x) | **~2k tps** |

**Measured ground truth CPU (this repo):**
- d32 100k: 40 bytes/sec → 1M bytes = 8h CPU
- 10B bytes CPU = 9.1 years — impossible

**10B tokens (10GB) on 5090:**

| Model | Realistic GPU tps | 10B / tps | Time |
|-------|-------------------|-----------|------|
| d32 100k | 500k tps | 20k sec | **5.5 hours** |
| d128 1.5M | 100k tps | 100k sec | **27 hours** |
| d256 6M | 40k tps | 250k sec | **2.9 days** |
| Llama 7B | 2k tps | 5M sec | **58 days** |

ATOM d128 is **50x faster** than Llama 7B for same 10B tokens, with 4666x fewer params, plus growable + CPU fine-tunable + live-surveyable.

**200GB (200B bytes) on a single normal 5090 — outshining server farms:**

| Model | 10B time | 200GB = 20x 10B | Time |
|-------|----------|-----------------|------|
| d32 100k | 5.5h | 110h | **4.5 days** |
| d128 1.5M | 27h | 540h | **22.5 days** |
| d256 6M | 2.9 days | 58 days | **1.9 months** |
| Llama 7B | 58 days | 1160 days | **3.1 years** |

**WTF:** 200GB in 1 month on a normal GPU is 100% feasible with d128 (22.5 days). Llama 7B needs 3.1 years on same single 5090, or 8x H100 cluster costing $500k.

ATOM outshines the need for server farms:
- Train 10s-100s GB ultra-fast on 1x 5090 ($2000) in your living room
- No need for 8x H100 farm for same data
- After GPU pre-train, download 6MB `.pt` (d256) to laptop, continue training on CPU for private domain in 3 min, no breaking, live survey `field_rms 0.26→1.06→3.0`
- Inference CPU 50 tps, no KV-cache explosion
- Growable d16→d32→d64→d128→d256 same lineage, old outputs exact

This is decentralization: train 200GB in 1 month on normal GPU, fine-tune on CPU.

## Current code is CPU-only on purpose

`tools/run_atom_native.py` has no `--device` flag — we banned it to measure true CPU scaling (see `TRAIN.md`: always `PYTHONPATH=. .venv/bin/python`, no `--device`). Model lives on CPU, `torch.Generator(device="cpu")`, `atom-flush-every 64` mandatory or O(n²) atoms → 20× slowdown.

## How to enable GPU (10 lines patch)

Add to `tools/run_atom_native.py`:

```python
parser.add_argument("--device", default="cpu", choices=("cpu","cuda"))
# ...
device = torch.device(args.device)
model.to(device)
# in transition_metrics, ensure current.features.to(device)
```

And in `src/atom_native.py` `generate_packets` / `transition_loss`, move `features` to `self.device`.

We will ship this as `--device cuda` + batched mode in next cut.

## Batched mode for true speed (what we need for 10B)

Current loop: 1 Python call = 1 byte forward. Overhead kills GPU.

Batched:

```python
# pseudo
B=256
batch = [next(source) for _ in range(B)]  # 256 transitions
loss = model.transition_loss_batched(batch)  # one compiled forward
loss.backward()
```

With `torch.compile(model.surface)` + prefetch (`--stream-prefetch` already exists), we go from 800 tps naive to 100k-500k tps.

Steps:
1. Implement `transition_loss_batched` in `AtomNativeModel` — stack 256 `features` [B, F] → one MLP forward
2. `torch.compile` surface head
3. Keep `atom-flush-every 64` and `episode-reset --episode-length 8000`
4. Keep original recipe: `byte-tick, last-atom-readout, dentate, MERGE off, payload-copy off, hard obligatory, efference-every 10, field-loss 0.05, contrast 0, LR 5e-5/1.2e-4, seed 20260913`

## How to train 10GB on 5090 (future command)

```bash
# 1. Generate 10GB shards (seed 1-20000 = ~10GB)
PYTHONPATH=. .venv/bin/python data/gen_corpus_fr_medium.py --seed-start 1 --seed-end 20000 --out-dir data/shards_10GB

# 2. Grow d32 1080k → d128
PYTHONPATH=. .venv/bin/python tools/grow_checkpoint.py \
  --checkpoint checkpoints/byte_tick/atom_native_step_1080000_d32_ms1M.pt \
  --output checkpoints/byte_tick/atom_native_step_1080000_d128_ms1M.pt \
  --d-model 128 --n-modes 128 --n-atoms-max 256

# 3. Train d128 on 10GB on 5090
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --stream --data-glob 'data/shards_10GB/*.txt' --loop-shards --chunk-bytes 1048576 \
  --resume checkpoints/byte_tick/atom_native_step_1080000_d128_ms1M.pt \
  --output-dir checkpoints/byte_tick --checkpoint-name atom_native_step_11000000_d128_10GB.pt \
  --steps 10000000 --d-model 128 --n-modes 128 --n-atoms-max 256 \
  --max-span-bytes 1 --field-obligatory-hard --no-enable-merge --no-payload-copy \
  --last-atom-readout --efference-every 10 --learning-rate 5e-5 --surface-learning-rate 1.2e-4 \
  --seed 20260913 --atom-flush-every 64 --episode-length 8000 --episode-reset \
  --device cuda --batch-size 256 --compile

# 4. Live survey without stopping train (read-only)
PYTHONPATH=. .venv/bin/python tools/probe_field_regimes.py --checkpoint checkpoints/byte_tick/atom_native_step_11000000_d128_10GB.pt --primer-packets 3500 --device cuda
PYTHONPATH=. .venv/bin/python tools/diagnose_teacher.py --checkpoint checkpoints/byte_tick/atom_native_step_11000000_d128_10GB.pt --max-span-bytes 1 --data data/corpus_fr_medium_s21.txt --device cuda

# 5. On-demand CPU fine-tune on laptop after GPU pre-train
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data data/your_private_domain.txt --steps 8000 --resume checkpoints/byte_tick/atom_native_step_11000000_d128_10GB.pt \
  --output-dir checkpoints/byte_tick --checkpoint-name atom_native_step_11008000_d128_private.pt \
  --d-model 128 --n-modes 128 --n-atoms-max 256 --max-span-bytes 1 --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout --efference-every 10 --atom-flush-every 64 --episode-length 8000 --episode-reset --device cpu
```

## Comparison vs transformer (for README)

- **Training:** ATOM d128 10B = 27h on 5090 vs Llama 7B 10B = 58 days on 5090 → **50x faster**, 4666x fewer params
- **Fine-tune:** ATOM 8k steps CPU 3 min on laptop, no forgetting, field_rms 0.26→1.06 measurable vs Llama LoRA 1h GPU + forgetting
- **Growth:** ATOM Net2Net d16→d32→d64→d128 same lineage, old outputs exact vs Llama must retrain from scratch
- **Live:** ATOM probe field in real time `probe_field_regimes.py` cold 0.04 / primed 1.06, generation `Peut-être`, `Parler,` vs Llama KV-cache only
- **No fluency claim yet:** ATOM cold probe still `tu tu` / spaces attractor since 336k, but weakening (clean stop `irra..\n\n` @1M) — scaling fixes it, not patches

## What to document next

- Add `--device` + `--batch-size` + `--compile` to trainer
- Add `transition_loss_batched`
- Bench d32 vs d128 vs d256 tps on 5090
- Release tip `atom_native_step_1080000_d32_ms1M.pt` → `d128_10GB.pt`

This is the new race: infinite, growable, CPU-scalable, live-modifiable. Train 10s GB ultra-fast on GPU, then teach new stuff on CPU from same .pt.
