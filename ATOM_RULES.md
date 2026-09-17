# ATOM — anti-drift rules

This document is the operational reference for the project. ATOM is not a
Transformer and must not be trained like a Transformer.

## Canonical pipeline

```text
TOKEN
  → TOROIDAL ATOM
  → PERSISTENT TOROIDAL FIELD
  → RK4 DYNAMICS
  → LOCAL TOROIDAL INTERACTION
  → AGGREGATION
  → ABSTRACTION
  → CONSOLIDATION
  → PRODUCTION / NEXT-TOKEN LOGITS
```

Each token is one system tick. For a sequence `x[0], ..., x[T-1]`, the learning
relation is:

```text
x[t] → model(x[t]) → logits[t] → target x[t+1]
```

The same model object is kept across ticks. Its field, atom collection,
abstractions, and persistent memory are not reset between tokens unless an
experience boundary explicitly requests a fresh state.

## Forbidden

1. **Flatten a sequence** with `reshape(-1)`, `flatten()`, or an equivalent
   operation before the model call.
2. Call `model(entire_sequence)` to produce all predictions for a sequence.
   The canonical path calls `model(row[t:t+1])` for each tick.
3. Replace RK4 dynamics, the toroidal field, or local interactions with a
   Transformer block, token–token attention, or another classic sequential
   architecture.
4. Add `nn.Transformer`, `TransformerEncoder`, `MultiheadAttention`, Q/K/V
   projections, or softmax attention inside `src/toroidal/`.
5. Reset toroidal state for every token or mini-batch without an explicit
   episode-boundary decision.
6. Use an external MLP/LSTM/GRU “thought” path to bypass toroidal dynamics.
   The learned encoder and production head are allowed as token→atom and
   state→logits translation; they do not replace the pipeline.
7. Claim a structural operation is executed when it is only predicted as a
   label. The current path creates a new atom per tick; MODIFY/MERGE/SPLIT
   must not be documented as active until they actually are.
8. Reintroduce a Hugging Face / GPT-2 tokenizer or a required `transformers`
   dependency on the default path. Canonical input is the Atomizer (UTF-8
   bytes). A tokenizer dependency must never justify adding attention to the
   model.
9. Rely on deleted benchmark or debug scripts. Historical comparisons do not
   define ATOM training.

## Allowed training path

The supported path is:

- `src/io/data.py` for data and sequences;
- `src/training/trainer.py` for the sequential loop;
- `src/toroidal/model.py` for the per-tick pipeline;
- `src/main.py` / `tools/run_atom_native.py` for the atom-native entry point.

The trainer accumulates losses over sequential transitions, runs a single
`backward()` per batch, then `optimizer.step()`. Checkpoints must keep weights
**and** dynamic state: field, atoms, history, abstractions, and persistent
memory.

## What ATOM actually is

ATOM may use learned parameters to map a token into physical atom properties
and to decode the current state into logits. That does not make ATOM a
Transformer. Computational memory is structured matter
(`r, φ, ω, E, κ, M, τ, ρ`), its spectral field, and its toroidal evolution.
