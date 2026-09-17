# Toroidal Fractal Intelligence (ATOM)

A novel AI architecture based on continuous structured learning through toroidal fractal dynamics.

## Vision

Build an AI architecture that doesn't fundamentally rely on the paradigm:

```
neurons → layers → weight matrices → massive backpropagation → fixed training
```

Instead, ATOM develops **dynamic computational matter** capable of continuously transforming tokens into internal structures, then evolving these structures through interaction, aggregation, abstraction, and consolidation.

## Core Principle

The system is based on continuous transformation:

```
TOKEN + CONTEXT + STATE → TOROIDAL STRUCTURE
```

The token is not simply transformed into activation and propagated through successive layers. It is converted into **computational matter**.

The atom-native refonte is documented in [`ATOM_NATIVE_TOKENIZATION.md`](ATOM_NATIVE_TOKENIZATION.md), and its CPU experiment is reported in [`ATOM_NATIVE_RUN.md`](ATOM_NATIVE_RUN.md). This path treats the input as a reversible stream observation and compiles it directly into contextual toroidal atoms rather than relying on the GPT-2 vocabulary.

This matter can then:
- Interact
- Superpose
- Aggregate
- Form structures
- Produce abstractions
- Be consolidated
- Be reused
- Contribute to output production

## Architecture

The non-negotiable ATOM invariants and prohibited patterns are documented in [`ATOM_RULES.md`](ATOM_RULES.md).


```
                TOKEN STREAM
                     │
                     ▼
          TOKEN → TOROIDAL MAP
                     │
                     ▼
          TOROIDAL ATOMS / FIELD
                     │
                     ▼
          FRACTAL SUPERPOSITION
                     │
                     ▼
              RK4 DYNAMICS
                     │
                     ▼
               INTERACTIONS
                     │
                     ▼
                AGGREGATION
                     │
                     ▼
                ABSTRACTION
                     │
                     ▼
               CONSOLIDATION
              ┌──────┴──────┐
              │             │
              ▼             ▼
         PERSISTENT      CURRENT
           STATE           STATE
              │             │
              └──────┬──────┘
                     ▼
                  OUTPUT
```

## Key Features

- **Infinite Learning**: No boundary between training and inference
- **Continuous State**: The toroidal field and structural memory persist across ticks
- **Structural Emergence**: Atoms → Aggregates → Abstractions
- **Selective Consolidation**: Only stable structures are persistent
- **Parameter Efficiency**: Shared rules control many structures
- **GPU/CPU Flexible**: Can run on modest hardware

## Installation

```bash
cd toroidal_fractal_intelligence
pip install -e .
```

## Quick Start

```bash
# Train
python -m src.main --mode train --max-steps 1000

# Chat
python -m src.main --mode chat --checkpoint checkpoints/final.pt

# Interactive mode
python -m src.main --mode interactive
```

## Testing

```bash
# Run the canonical invariant tests
python -m pytest test/core_invariants.py
```

## Project Structure

```
toroidal_fractal_intelligence/
├── src/
│   ├── toroidal/           # Core architecture (10 modules)
│   │   ├── encoder.py      # Token → Toroidal Atom
│   │   ├── atom.py         # Atom data structures
│   │   ├── state.py        # Fractal superposition
│   │   ├── dynamics.py     # RK4 integration
│   │   ├── interaction.py  # Pairwise interactions
│   │   ├── aggregation.py  # Hierarchical aggregation
│   │   ├── abstraction.py  # Pattern extraction
│   │   ├── consolidation.py # Persistent memory
│   │   ├── production.py   # Output generation
│   │   └── model.py        # Complete model
│   ├── io/                 # Data loading
│   ├── evaluation/         # Metrics
│   ├── training/           # Training loop
│   └── main.py             # CLI entry point
├── checkpoints/            # Model checkpoints (.pt)
├── results/                # Experiment results
├── *.py                    # Test scripts
└── *.md                    # Documentation
```

## Research Questions

1. **Structural Efficiency**: Can we get more capacity per parameter through structure composition?
2. **Infinite Learning**: Can the system learn continuously without forgetting?
3. **Computational Economy**: Can shared rules control many structures efficiently?
4. **Abstraction Reuse**: Can learned abstractions transfer across domains?

## Experimental Results

### Tiny Shakespeare (500 steps)
- Parameters: 3,718
- Final atoms: 87
- Final loss: 8.12 (from 10.5)
- Atoms/parameter: 0.0234

### Benchmark (100 steps)
- ATOM: 478,534 params, 100 atoms, loss=6.22
- Transformer: 41,044 params, 0 structures, loss=6.29
- ATOM creates structural capacity Transformer cannot match

## Hypotheses

1. **Structural Efficiency Hypothesis**: Capacity can grow through structure composition without linear parameter increase
2. **Infinite Learning Hypothesis**: Continuous learning without catastrophic forgetting is possible
3. **Computational Economy Hypothesis**: Shared rules can control many structures efficiently

## Citation

```bibtex
@misc{toroidal_fractal_intelligence,
  title={Toroidal Fractal Intelligence: A Continuous Structured Learning Architecture},
  author={PHIL},
  year={2026},
  version={0.1.0}
}
```

## License

Proprietary — For research and evaluation purposes only.

## Status

**Experimental Research System**

The architecture is implemented and functional. Basic proofs of concept are demonstrated. Rigorous benchmarking requires:
- Fixed parameter budget comparison
- Multiple random seeds
- Real-world datasets (WikiText-2, Wikipedia)
- Extended training (1000+ steps)

## Next Steps

1. Run full benchmark with WikiText-2
2. Test scaling to larger configurations
3. Compare with equivalent Transformer
4. Publish scaling laws paper

---

**Built with passion for alternative AI architectures.**
