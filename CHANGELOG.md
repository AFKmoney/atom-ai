# Changelog — atom-ai / ATOM

All dates in **America/Vancouver (PT)**.

## 2026-09-17 — atom-ai clean export `0.2.0`

- New git-ready tree branded **atom-ai** (package `atom-ai`, imports still `src.*`).
- Spine only: Atomizer, AtomNative, toroidal field, tools, tests, bilingual docs.
- Quarantine GPT-2 / docs-slop **excluded**.
- Shipped sample: `checkpoints/atom_native_chat_persist/` (~8 MB, ~1.05M steps).
- Tiny dialogue samples under `data/samples/`.
- Docs: README, MANIPULATIONS, ARCHITECTURE, TRAINING, CHAT, ANTISLOP, plus
  GEN_DEBUG / DECAY_FIX / REPAIR_AND_TRAIN / ATOM_RULES / ATOM_NATIVE_TOKENIZATION.
- `pyproject.toml`: requires `torch`, `numpy` only (no transformers).

## 2026-09-17 — energy_decay clamp

- Always-on bounds **`[0.3, 0.95]`** on every step and on checkpoint load.
- Documented in `DECAY_FIX.md`; related `field_max_rms=3.0` / surface LR notes.

## 2026-09-17 — persist train ~1.05M

- `--no-episode-reset`, `--atom-flush-every 256`, dialogue corpus.
- `BEST_CHAT` → `atom_native_chat_persist/atom_native.pt`.

## 2026-09-17 — generation repair + continued train

- Surface `LayerNorm`, bias damp on legacy load, boundary inference, episode 512.
- Smoke tests + `smoke_gen_check.py`.
- +250k steps → ~650k total; prompt-diverse printable gens; chat still embryonic.
- See `GEN_DEBUG.md`.

## 2026-09-16 — quarantine & atom-native default

- GPT-2 tokenizer/loaders/token-id trainer quarantined.
- Docs slop quarantined.
- CLI + tools default to Atomizer path.
- Dialogue corpus built; first ~80k CPU chat train.
- See `REPAIR_AND_TRAIN.md`.

## 2026-09-13 — atom-native tokenization experiment

- Atomizer + AtomCompiler design (`ATOM_NATIVE_TOKENIZATION.md`).
