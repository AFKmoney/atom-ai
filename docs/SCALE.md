# SCALE — GPT-3 capability without GPT-3 parameter/GPU-farm scale

## Thesis

ATOM does **not** aim to match GPT-3’s ~175B parameters or its training-cluster
footprint. The goal is comparable **useful intelligence** — coherent continuum
behavior over language — via a different capacity substrate:

| GPT-3 world | ATOM world |
|-------------|------------|
| ~175B shared weights | Small shared rule set Θ (field / dynamics / surface) |
| GPU farms, huge batches | CPU / single-box friendly continuum |
| Capacity ≈ parameter count | Capacity ≈ **persistent structured matter** |
| One-shot dense pretrain | Infinite resume + consolidation + streaming ingest |

Parameters stay modest. What grows is the **toroidal field**, structural atoms,
aggregates, abstractions, and consolidation — computational matter that
persists across ticks and runs.

## Why streaming ingest matters

A full corpus must not be forced into a giant in-RAM packet list. Streaming
full-dataset ingest (`src/io/stream_corpus.py`, `--stream --data-glob`) reads
UTF-8-safe shards online, atomizes with `encode_bytes(..., flush=False)`,
flushes at shard ends, and yields `(current, target)` transitions forever
(`--loop-shards`). Complete datasets become train-able on one machine without
matching GPT-3’s hardware narrative.

## Capacity story (honest)

- **Not claimed:** “we beat GPT-3 on every benchmark with 64 modes.”
- **Claimed:** the architectural bet is that structured persistence + streaming
  complete data can approach *capability* without *parameter/GPU-farm* scale.
- Metrics (byte CE, packet ppl) are training signals, not chat-quality proofs.
  Judge generations and long-horizon field behavior separately.

## Practical scale path

1. Shard corpora under `data/shards/*.txt` (or any glob).
2. Train with `--stream --data-glob ... --loop-shards --no-episode-reset`.
3. Resume forever; let consolidation accumulate; soft-flush atoms, not the field.
4. Grow Θ only when dynamics demand it — not to chase 175B vanity counts.

See also: `docs/TRAINING.md`, `ATOM_RULES.md`, `ATOM_NATIVE_TOKENIZATION.md`.


## Persistence test (field farm vs CE-only)

CE down + fluent-looking metrics **without** field reconstruction still means
**Transformer regime without the farm.** Capacity that matters for ATOM is
persistent structured matter in the toroidal field, not parameter count alone.

Latest probe (`tools/probe_field_persistence.py`):

- Mean RMS after prompt / after 20 gen: 0.003336 / 0.022708
- Off-diag cosine surface logits / alpha: 0.866300 / 0.713154
- Verdict: field is carrying structure (prompt-sensitive; diagnostic reconstruction partial)

See `docs/PERSISTENCE_TEST.md` and `docs/FIELD_PROBE.md`.
