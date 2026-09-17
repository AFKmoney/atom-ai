# Antislop — what was rejected and why / Ce qui a été rejeté

**EN** + **FR**

---

## English

### Rejected code paths

| Item | Why rejected |
|------|----------------|
| GPT-2 / HF `AutoTokenizer` default | Reintroduces BPE vocab as fundamental unit; contradicts atom-native |
| `legacy_gpt2_data.py` WikiText BPE loaders | Couples training to Transformers stack |
| `legacy_token_id_trainer.py` as default train | Token-ID CE trainer ≠ packet/tick ATOM path |
| Any `nn.Transformer` / `MultiheadAttention` / QKV | Forbidden by `ATOM_RULES.md`; not ATOM |
| Flatten-sequence single forward | Breaks tick dynamics and persistent field |
| Shipping 40MB+ weight dumps “for convenience” | Prefer train docs + one ≤10MB useful sample |
| Hardcoded `transformers` in required deps | Default install must be torch+numpy only |

### Rejected documentation / process

| Item | Why |
|------|-----|
| Duplicate RESULT / WORK_LOG / SCALING_* mountains | Noise competing with spine docs |
| Benchmark marketing vs Transformer as success criterion | ATOM success ≠ beating GPT on CE |
| Claiming MODIFY/MERGE/SPLIT as active when only labels | Honesty rule in ATOM_RULES |
| Claiming fluent chat from byte-LM smoke | Metrics honesty |

### Kept despite temptation

- Toroidal `encoder` + `production` heads (translation in/out — **not** attention).
- Optional higher surface LR (still toroidal field learning).
- Episode soft atom flush (performance) **without** pretending the field was wiped
  when `--no-episode-reset` is on.

### Export exclusions (this repo)

- `_quarantine_transformer_slop/**`
- `_quarantine_docs_slop/**`
- `agents/` (empty / `__pycache__` only)
- `.venv/`, `*.egg-info/`, `__pycache__/`
- Huge `data/corpus_*.txt`, `data/raw_gutenberg/`, `data/train.txt`
- `logs/**`, `results/**` noise
- Extra checkpoints beyond the one BEST persist sample

---

## Français

### Rejeté

Tokenizer GPT-2 / HF par défaut, chargeurs BPE, trainer token-ID comme chemin
principal, tout bloc Transformer/attention, aplatissement de séquence, deps
`transformers` obligatoires, dumps de poids énormes, docs RESULT/WORK_LOG
bruyantes, marketing « on bat le Transformer », prétention de chat fluide.

### Conservé

Cœur toroïdal, Atomizer, outils atom-native, tests smoke, un checkpoint démo
~8 Mo, documentation bilingue honnête.
