# Data

Shipped samples (small, for smoke / demo):

| Path | Approx size | Role |
|------|-------------|------|
| `samples/dialogue_tiny.txt` | ~12 KB | Tiny FR/EN dialogue seed |
| `samples/corpus_train_dialogue_slice.txt` | ~80 KB | Head slice of the dialogue training corpus |
| `CORPUS_*.txt` | meta | How the full corpora were built in the repair session |

**Not shipped:** multi-MB WikiText mixes, raw Gutenberg dumps, 13 MB mixed corpora. Rebuild locally (Project Gutenberg + any UTF-8 dialogue you own) and point `tools/run_atom_native.py --data ...` at them. See `docs/TRAINING.md` and `docs/REPAIR_AND_TRAIN.md`.

Format tip: multi-turn lines like `Utilisateur:` / `Assistant:` help the surface head learn dialogue-ish fragments.
