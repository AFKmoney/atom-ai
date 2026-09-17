> **atom-ai export note:** Quarantine folders and multi-MB corpora from the
> original `toroidal-fractal-intelligence` repair tree are **not** shipped here.
> Paths below describe historical work; use `data/samples/` and rebuild corpora locally.

# Repair and train — atom-native restoration

Date: 2026-09-16 (PT)

## Quarantined

### `_quarantine_transformer_slop/`
| Item | Why |
|------|-----|
| `tokenizer.py` | HF `AutoTokenizer` / GPT-2 default |
| `legacy_gpt2_data.py` | WikiText loaders that BPE-tokenize with GPT-2 |
| `legacy_token_id_trainer.py` | Snapshot of token-ID CE trainer (live copy kept for resume tests only) |

See that folder’s `README.md`.

### `_quarantine_docs_slop/`
Noisy duplicate result/work-log markdown that competed with the spine docs.

## Fixed / promoted

- **Default CLI** `src/main.py` → atom-native `train` / `chat` / `interactive` (no GPT-2).
- **Chat** `tools/chat_atom_native.py` — load checkpoint, Atomizer encode, `generate_packets`, UTF-8 decode.
- **Train** `tools/run_atom_native.py` — checkpoint name `atom_native.pt`, `--resume`, random episode starts, FR prompts.
- Docs: `README.md`, `ARCHITECTURE.md`, `USAGE.md`, `GETTING_STARTED.md` point to Atomizer.
- `pyproject.toml` — atom-native entry points; HF/transformers optional as `legacy-hf`.
- Toroidal core untouched; no attention added.

## Corpus

| Path | Size | Notes |
|------|------|-------|
| `data/corpus_mixed_fr_en.txt` | ~13.4 MB | Full mix: WikiText-2 train + Voltaire Candide FR + Perrault Contes + FR/EN chat seed |
| `data/corpus_train_chat.txt` | ~778 KB | Training slice (RAM-safe atomization); EN WikiText head + FR PD excerpts + chat seed |
| `data/CORPUS_INFO.txt` | — | Pointers |

Sources need no auth (GitHub raw WikiText + Project Gutenberg).

## Training config (box / CPU)

```text
d_model=64  n_modes=64  n_atoms_max=512  max_span_bytes=16
episode_length=64  field_max_rms=4.0  lr=1e-3  steps=80000
output: checkpoints/atom_native_chat/atom_native.pt
```

Continue:

```bash
cd /workspace/repos/toroidal-fractal-intelligence
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data data/corpus_train_chat.txt \
  --output-dir checkpoints/atom_native_chat \
  --resume checkpoints/atom_native_chat/atom_native.pt \
  --steps 80000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --field-max-rms 4.0
```

Or train on the full 13 MB corpus if you have enough RAM for atomization.

## Chat

```bash
PYTHONPATH=. .venv/bin/python tools/chat_atom_native.py \
  --checkpoint checkpoints/atom_native_chat/atom_native.pt \
  --prompt "Bonjour, comment ça va"

PYTHONPATH=. .venv/bin/python -m src.main --mode interactive \
  --checkpoint checkpoints/atom_native_chat/atom_native.pt
```

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest test/ -q
# Expected: 10 passed
```

## Honesty

Conversational quality after one CPU session will still be rough: the model predicts next **byte packets**, not chat-tuned LLM replies. Expect noisy UTF-8 early; longer training + more FR dialogue data improves surface text. This restores the **atom-native spine**, not GPT-2 parity.


## Training result (this session)

- **Completed**: 80 000 steps on CPU (~670 s wall, ~119 transitions/s)
- **Checkpoint**: `checkpoints/atom_native_chat/atom_native.pt` (~6.8 MB)
- **Loss**: first ≈ 8.38 → final train step ≈ 7.90; validation loss ≈ 5.89; val byte-ppl ≈ 29.9
- **Finite**: parameters and losses stayed finite; reload generation matched

### Sample dialogue (still rough — byte-packet LM, not chat-tuned)

```
Vous: Bonjour, comment ça va
ATOM:  ur hs e tnr
et n@t,.nt  eeein ...

Vous: The future of AI is
ATOM: ,,hstest a aoa at   int<e i et ersa r> ...
```

Surface text is mostly Latin letters/spaces with noise. Continue training or enlarge the FR dialogue seed for better chat.

### Exact chat commands

```bash
cd /workspace/repos/toroidal-fractal-intelligence
PYTHONPATH=. .venv/bin/python tools/chat_atom_native.py \
  --checkpoint checkpoints/atom_native_chat/atom_native.pt \
  --prompt "Bonjour, comment ça va"

PYTHONPATH=. .venv/bin/python -m src.main --mode interactive \
  --checkpoint checkpoints/atom_native_chat/atom_native.pt
```


## Gen-fix + continued train (2026-09-17 PT)

See **GEN_DEBUG.md** for root causes (surface bias drowning field; episode resets).

Key training defaults now:
- `episode_length=512` (was 64)
- `--no-episode-reset` by default (field persists; soft atom-list clear near cap)
- Surface `LayerNorm` + legacy bias damp on load

Resume (this session target: +300k on dialogue corpus):

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data data/corpus_train_dialogue.txt \
  --output-dir checkpoints/atom_native_chat \
  --resume checkpoints/atom_native_chat/atom_native.pt \
  --steps 300000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 --episode-length 512 --no-episode-reset \
  --field-max-rms 4.0 --learning-rate 3e-4 --log-every 100
```

Smoke:

```bash
PYTHONPATH=. .venv/bin/python tools/smoke_gen_check.py \
  --checkpoint checkpoints/atom_native_chat/atom_native.pt
```


## Gen-fix session outcome (2026-09-17 PT)

- Docs: `GEN_DEBUG.md`
- Tests: 13 passed (`test/test_generation_smoke.py` added)
- Train: +250k steps → **650k total** on `data/corpus_train_dialogue.txt`
- Log: `logs/train_after_genfix_250k.log` (also `logs/train_after_genfix_20260917.log`)
- Checkpoint: `checkpoints/atom_native_chat/atom_native.pt` (BEST_CHAT updated)
- Val byte_ppl ≈ 31; gens prompt-sensitive + printable; FR chat still embryonic (`Je…` fragments, not coherent replies)
