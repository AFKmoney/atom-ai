# Architecture — ATOM / atom-ai

### Pipeline (canonical)

```text
UTF-8 text
  → Atomizer.encode()          # structural byte spans, 320-D features
  → AtomPacket(payload, features, boundary, …)
  → AtomCompiler               # features → ToroidalAtom properties
  → ToroidalFractalIntelligence tick:
        encoder/atom inject → spectral field
        → RK4DynamicsEngine
        → ToroidalInteraction (local)
        → AggregationEngine
        → AbstractionEngine
        → ConsolidationEngine
        → production / field readout
  → AtomSurfaceHead            # LayerNorm → linear → byte logits / packet decode
  → UTF-8 bytes (next packet)
```

One **packet** ≈ one **tick**. The same model object keeps field α, atom list
(within flush policy), abstractions, and consolidation across ticks unless an
episode boundary explicitly resets.

### Modules

| Path | Role |
|------|------|
| `src/io/atomizer.py` | No BPE / no HF. Reversible UTF-8 spans + context features |
| `src/atom_native.py` | `AtomNativeModel`, surface head, energy_decay clamp, generate |
| `src/toroidal/*.py` | Atom, state/field, RK4, interaction, aggregation, abstraction, consolidation, production, model |
| `tools/run_atom_native.py` | Train loop: random episode starts, optional field persist |
| `tools/chat_atom_native.py` | Load ckpt → prime prompt → `generate_packets` |
| `tools/smoke_gen_check.py` | Prompt diversity / printable UTF-8 smoke |

### What is *not* in the graph

- `nn.Transformer` / `MultiheadAttention` / QKV softmax attention
- Hugging Face `AutoTokenizer` / GPT-2 vocab on the default path
- Flattening a whole sequence into one model call

See `ATOM_RULES.md` (repo root and `docs/`).

### State & persistence

- **Field**: spectral superposition (`n_modes × d_model`), RMS capped (`field_max_rms`)
- **Atoms**: explicit collection up to `n_atoms_max`; soft flush without wiping field when `--no-episode-reset`
- **Consolidation**: long-lived structural memory
- **Checkpoints**: weights **and** dynamic state (field, atoms, training step)

### Surface head notes

Post gen-fix (2026-09-17): `LayerNorm` before surface linears; legacy bias damp
on load; printable soft bias; structural boundary inference for generated
payloads (`GEN_DEBUG.md`).
