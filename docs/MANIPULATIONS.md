# Manipulations — chronological repair log

Session work that produced the clean **atom-ai** spine from
`toroidal-fractal-intelligence` (Grok / repair agents, America/Vancouver PT).

---

## 0. Context

Starting point: a toroidal research repo that had drifted toward GPT-2
tokenization, HF loaders, noisy benchmark/work-log markdown, and a weak
default “chat” path. Goal: restore **atom-native** as the only default,
document honestly, train on dialogue, fix generation bugs, persist training.

---

## 1. Study arena branch & quarantine (2026-09-16 PT)

1. Inspected arena / local tree; identified GPT-2 `tokenizer.py`,
   `legacy_gpt2_data.py`, `legacy_token_id_trainer.py` as non-spine.
2. Moved them to `_quarantine_transformer_slop/` (excluded from **atom-ai** export).
3. Moved duplicate RESULT / WORK_LOG / SCALING mountains to
   `_quarantine_docs_slop/` (also excluded here).
4. Confirmed toroidal core had **no** `nn.Transformer` / `MultiheadAttention`.
5. Promoted CLI defaults: `src/main.py` → atom-native train/chat/interactive.
6. `pyproject.toml`: deps = `torch`, `numpy`; HF optional only as `legacy-hf`
   (removed entirely in this **atom-ai** export).

---

## 2. Atom-native default & tools (2026-09-16 PT)

- `tools/run_atom_native.py` — Atomizer corpus train, checkpoint `atom_native.pt`, `--resume`.
- `tools/chat_atom_native.py` — load → encode prompt → `generate_packets` → UTF-8.
- Docs rewritten toward Atomizer (`ATOM_NATIVE_TOKENIZATION`, README, ARCHITECTURE).
- Tests: `test/test_atom_native.py`, `test/core_invariants.py`, resume tests kept.

---

## 3. Dialogue corpus (2026-09-16 PT)

- Built FR-heavy dialogue mix: synthetic `Utilisateur:`/`Assistant:` turns +
  public-domain French theatre/fables (Corneille, Molière, Racine, La Fontaine,
  Hugo, Voltaire Candide, Perrault) + existing FR/EN chat seed.
- Training slice `corpus_train_dialogue.txt` (~2.8 MB) used for long runs;
  full mix archived locally (~4 MB dialogue + larger mixed files).
- **atom-ai export** ships only tiny samples (`data/samples/*`) + corpus info
  pointers — not the multi-MB downloads.

---

## 4. First CPU train (~80k steps) (2026-09-16 PT)

- Config: `d_model=64`, `n_modes=64`, `n_atoms_max=512`, `episode_length=64` (later fixed).
- Checkpoint `atom_native_chat/atom_native.pt`; loss moved but chat was rough
  byte-LM noise — documented honestly in `REPAIR_AND_TRAIN.md`.

---

## 5. Generation bugs diagnosed (2026-09-16 → 17 PT)

Root causes (`GEN_DEBUG.md`):

1. **Surface bias drowned the field** — `|W x|/|b| ≈ 0.005`; all prompts greedily
   decoded to the same first packet (`\ne  `); byte-logit cosine ≈ 1.0 across prompts.
2. **Episode reset every 64 packets** — field/consolidation wiped; dialogue never persisted.
3. **`boundary="generated"`** on feedback packets — feature code unseen in train.
4. Sampling noise amplified a collapsed head (secondary).

**Fixes:**

- `LayerNorm` before surface linears; legacy load damp biases `×0.05`.
- Infer structural boundary on generated payloads.
- Printable soft bias; safer sampling defaults.
- Default `episode_length=512`; `--no-episode-reset` (BooleanOptional).
- Soft atom-list flush near cap without wiping field.
- Cumulative `training.step` on resume.
- Role priming in chat; `test/test_generation_smoke.py` + `tools/smoke_gen_check.py`.

---

## 6. Continued train after gen-fix (~650k total) (2026-09-17 ~02:00 PT)

- +250k steps on dialogue corpus; migration loss bump then recovery.
- Val byte_ppl ~31; gens prompt-sensitive + printable; FR chat still embryonic.
- Discovered `energy_decay` had drifted badly (e.g. negative / near-zero) → clamp work.

---

## 7. Persist training to ~1.05M+ (2026-09-17 PT)

- Long persist run: `--no-episode-reset`, `--atom-flush-every 256`,
  `episode_length=512`, dialogue corpus.
- Checkpoint: `checkpoints/atom_native_chat_persist/atom_native.pt` (**~8 MB**),
  `total_steps=1050000` — marked in `BEST_CHAT.txt`.
- Metrics honesty: loss/ppl ≠ chat quality; recorded `energy_decay_bounds: null`
  on that run (pre-clamp defaults).

---

## 8. Energy decay clamp (2026-09-17 PT)

(`DECAY_FIX.md`)

- After persist, `energy_decay ≈ 0.001` → near-total wipe each tick
  (`decay = (energy_decay - 1) * alpha`).
- Always-on default band **`[0.3, 0.95]`** every optimizer step **and** on `load()`.
- Retrain path toward `atom_native_chat_persist_v2` with `field_max_rms=3.0`
  and surface LR group (may be incomplete in source; code clamp is what matters).

---

## 9. Clean export → GitHub `atom-ai` (2026-09-17 PT)

This tree (`/workspace/exports/atom-ai/`):

- Includes only the good spine (atom-native + toroidal + tools + tests + docs).
- Excludes quarantine dirs, GPT-2 paths, `__pycache__`, egg-info, `.venv`,
  huge raw corpora, noisy logs/results mountains, empty `agents/`.
- Brands package **`atom-ai`**; keeps `src.*` imports.
- Ships one useful ~8 MB persist ckpt + tiny data samples.
- Documentation is **English-only** (no bilingual FR sections).
- Parent agent pushes to GitHub; this export does **not** push.

---

## Timeline cheat-sheet

| When (PT) | What |
|-----------|------|
| 2026-09-16 | Quarantine GPT-2 + docs slop; atom-native CLI; corpus; ~80k train |
| 2026-09-16/17 | Gen debug: surface bias, episode 64, boundary |
| 2026-09-17 ~02:00 | +250k → ~650k; smoke gen improved diversity |
| 2026-09-17 | Persist → **1.05M**; BEST_CHAT → persist ckpt |
| 2026-09-17 | Decay clamp `[0.3, 0.95]`; field_max_rms 3.0 notes |
| 2026-09-17 | Build clean `/workspace/exports/atom-ai/` for GitHub |
