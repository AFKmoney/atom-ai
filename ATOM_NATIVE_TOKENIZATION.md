# ATOM-native atomization — `signal → atom` redesign

Experiment date: 2026-09-13.

## Intent

The existing GPT-2 path was still classic tokenization:

```text
text → GPT-2 ID → static embedding → mapping → atom
```

That path keeps a 50,257-class vocabulary and treats the ID as the fundamental
unit. It does not match the ATOM document intent:

```text
T(x, c, S) → A
```

This redesign therefore introduces an **atomizer**, not a smaller subword
vocabulary:

```text
raw UTF-8 stream
  → byte observations and local buffer
  → reversible, contextual atomic packet
  → eight ToroidalAtom properties
  → persistent field / RK4 / interactions / aggregation / abstraction /
    consolidation
```

The byte is an input observation, not a semantic table entry. The atomic packet
is the unit that enters the toroidal core.

## Implementation

### `src/io/atomizer.py`

`Atomizer` segments the stream online, without BPE, SentencePiece, or Hugging
Face. Each `AtomPacket` contains:

- raw UTF-8 `payload`, kept as bytes for reversibility;
- `start` and `end` offsets;
- the boundary that triggered emission;
- a structural level;
- fragment phase and duration;
- a 320-D deterministic observation vector.

The 320 observations are:

```text
256 byte histograms
+ 16 first-nibble groups
+ 16 last-nibble groups
+ 16 local statistics
+ 16 sliding-context values
```

Statistics include length, whitespace/digit/letter/punctuation/control ratios,
entropy, class transitions, phase, previous length, and similarity to previous
context.

The same surface fragment can therefore yield different observations depending
on the preceding stream. Context is serializable with the atomizer.

Current boundaries are structural and deterministic:

- whitespace and newlines;
- punctuation;
- letter/digit transitions;
- duration limit `max_span_bytes`;
- end of stream.

The duration limit does not split a UTF-8 code point. Concatenating payloads
reproduces the original bytes exactly.

### `src/atom_native.py`

`AtomCompiler` takes the 320 observations and produces the eight properties
directly:

```text
(r, phi, omega, E, kappa, M, tau, rho)
```

There is no `Embedding(vocab_size=50257)` on this path.

`AtomNativeModel` reuses existing toroidal modules without modifying
`src/toroidal/` in this redesign:

- `FractalSuperpositionState`;
- `RK4DynamicsEngine`;
- `ToroidalInteraction`;
- `AggregationEngine`;
- `AbstractionEngine`;
- `ConsolidationEngine`.

The older encoder and GPT-compatible head remain in the core checkpoint for
compatibility, but are frozen and unused by the atom-native experiment.

### Atomic output

To reconstruct surface text without 50,257 logits, the atom-native head predicts
the next packet as:

```text
1..max_payload_bytes
+ 256 byte classes per position
```

Transition loss is:

```text
loss = cross_entropy(next packet length)
     + cross_entropy(next packet bytes)
```

In the experiment, `max_payload_bytes=16`. One transition is therefore one
atomic packet, not one GPT-2 token.

## Preserving the ATOM core

No Transformer, attention mechanism, Q/K/V, flattened sequence, or per-tick
reset was added. The packet path is:

```text
AtomPacket
  → AtomCompiler
  → ToroidalAtom
  → persistent field
  → RK4
  → toroidal interaction
  → aggregation / abstraction / consolidation
  → AtomSurfaceHead
```

The reset used in the early benchmark occurs only between explicit 64-packet
episodes to bound the structural collection during CPU measurement. It is not
a reset between transitions inside the same episode.

## Tests added

`test/test_atom_native.py` covers:

1. exact UTF-8 reconstruction;
2. observation dimension and non-ID nature;
3. dependence of the same fragment on prior context;
4. exact atomizer state restore;
5. model forward, loss, backward, and finiteness;
6. atom-native checkpoint save/reload.

The existing core is also checked by `test/core_invariants.py`.

## Known limits

This first redesign is an atom-native research path, not yet a proof of general
linguistic quality:

- boundaries are deterministic; they are not yet learned by the field;
- local data was a 3,070-character WikiText-2 raw excerpt;
- smoke dimensions remain `d_model=8`, `n_modes=8`;
- the surface model trains on bounded byte packets;
- generation after 500 transitions remains experimental and partly incoherent;
- the explicit atom collection still needs study on much longer streams.

Full run results lived in `ATOM_NATIVE_RUN.md` and
`checkpoints/atom_native_500/` in the source tree (not necessarily shipped in
this export).
