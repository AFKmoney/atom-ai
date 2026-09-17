# Toroidal Fractal Intelligence — Architecture Documentation

## Overview

This document describes the complete architecture of the Toroidal Fractal Intelligence system, a novel AI architecture based on continuous structured learning rather than traditional neural network paradigms.

## Core Philosophy

**Traditional AI**: `neurones → couches → matrices de poids → backpropagation massive → entraînement fini`

**Toroidal Fractal AI**: `TOKEN → ATOME TOROÏDAL → SUPERPOSITION FRACTALE → DYNAMIQUE → AGGRÉGATION → ABSTRACTION → CONSOLIDATION → SORTIE`

The key insight: intelligence emerges from the *dynamics and organization of structures*, not just from static weights.

The non-negotiable implementation rules are in [`ATOM_RULES.md`](ATOM_RULES.md). In particular, ATOM is stateful and tick-based; it is not a Transformer and its token stream must never be flattened into one model call.

## Mathematical Foundation

### Toroidal Atom

The fundamental building block is the **toroidal atom**:

```
A_i = (r_i, φ_i, ω_i, E_i, κ_i, M_i, τ_i, ρ_i)
```

Where:
- `r_i` = position/rayon in latent space
- `φ_i` = phase
- `ω_i` = natural frequency
- `E_i` = energy/activation/importance
- `κ_i` = coupling strength
- `M_i` = local memory
- `τ_i` = time scale
- `ρ_i` = hierarchical level

### Token → Atom Conversion

The central function:

```
T(x, c, S) → A
```

Where:
- `x` = token
- `c` = context
- `S` = current state of computational matter
- `A` = toroidal atom or structural modification

The encoder returns an operation label, but the current forward path creates one new atom per tick. `MODIFY`, `MERGE`, `REINFORCE`, `SPLIT`, `ABSTRACT`, and `CONSOLIDATE` are not active structural mutations and must not be documented as implemented mechanisms.

### Fractal Superposition

```
S = Σ_i α_i A_i
```

The spectral field is the compact operational state with `O(n_modes)` size. The model also keeps an explicit atom collection as structural memory, so total memory is not purely `O(n_modes)` in the current implementation.

### RK4 Dynamics

The superposition evolves via differential equation:

```
dS/dt = F(S, x, C, Θ)
```

Integrated with Runge-Kutta 4th order:
```
k1 = F(S_t)
k2 = F(S_t + dt/2 * k1)
k3 = F(S_t + dt/2 * k2)
k4 = F(S_t + dt * k3)
S_{t+dt} = S_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

## Module Architecture

```
toroidal_fractal_intelligence/
├── src/
│   ├── toroidal/
│   │   ├── encoder.py       # Token → Atom conversion
│   │   ├── atom.py          # Atom data structures
│   │   ├── state.py         # Fractal superposition state
│   │   ├── dynamics.py      # RK4 integration engine
│   │   ├── interaction.py   # Pairwise interactions
│   │   ├── aggregation.py   # Structure aggregation
│   │   ├── abstraction.py   # Pattern abstraction
│   │   ├── consolidation.py # Persistent memory
│   │   ├── production.py    # Output generation
│   │   └── model.py         # Complete model integration
│   ├── io/
│   │   ├── tokenizer.py     # Text tokenization
│   │   └── data.py          # Data loading utilities
│   ├── evaluation/
│   │   └── metrics.py       # Performance metrics
│   ├── training/
│   │   └── trainer.py       # Training loop
│   └── main.py              # Entry point
├── checkpoints/             # Model checkpoints (.pt)
├── logs/                    # Training logs
└── results/                 # Evaluation results
```

## Design Decisions

### 1. Spectral Field Representation (state.py)

**Decision**: Use a fixed number of spectral modes (256) for the operational field while retaining explicit atoms as structural memory.

**Rationale**: 
- O(modes) storage vs O(N * d_model) for explicit atoms
- Natural frequency/phase separation
- Efficient RK4 integration on the field
- Capacity grows with structure composition, not parameter count

**Trade-off**: Some information loss in projection, but regained through hierarchy.

### 2. Shared Parameters Theta (dynamics.py)

**Decision**: A small set of shared parameters controls interactions for all atoms.

**Rationale**:
- Decouples capacity from training cost
- Enables billions of potential states with modest parameter count
- Follows the principle: `capacity >> independent parameters`

**Implementation**: `coupling_scale`, `energy_decay`, and `phase_sync` in the toroidal dynamics. These are shared rules, not per-token or per-atom attention parameters.

### 3. Sparse Interaction (interaction.py)

**Decision**: Use local toroidal neighbour interactions over the spectral field. The interaction uses periodic rolls, distance/phase coupling and field gradients; it has no Q/K/V projections or attention softmax.

**Rationale**:
- Computational efficiency for large N
- Biological plausibility (local connectivity)
- Still captures global structure through field

### 4. Hierarchical Aggregation (aggregation.py)

**Decision**: Allow recursive aggregation up to depth 5.

**Rationale**:
- Enables `atom → motif → structure → meta-structure → abstraction`
- Capacity grows recursively
- Mirrors biological hierarchy (neuron → column → cortex)

### 5. Dynamic Consolidation (consolidation.py)

**Decision**: Only consolidate stable, useful structures.

**Rationale**:
- Saves computation on transient structures
- Creates persistent memory for important patterns
- Enables infinite learning without unbounded growth

### 6. Continuous Training Loop (trainer.py)

**Decision**: No training/inference boundary; infinite data stream. The canonical trainer processes one token transition per tick and preserves the model state between ticks.

**Rationale**:
- True continuous learning
- No catastrophic forgetting (structures consolidate)
- Real-time adaptation

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Hidden dimension |
| `n_modes` | 256 | Number of spectral modes |
| `n_atoms_max` | 1024 | Maximum concurrent atoms |
| `dt` | 0.01 | RK4 time step |
| `n_steps` | 4 | RK4 integration steps |
| `phase_coherence_threshold` | 0.7 | Aggregation threshold |
| `energy_threshold` | 0.3 | Minimum energy for consolidation |
| `consolidation_threshold` | 0.7 | Stability threshold |

## Learning Efficiency Metric

The fundamental metric:

```
Learning Efficiency = Information Acquired / Computational Cost
```

Measures:
- `atoms_per_parameter`: Structural capacity per trainable parameter
- `information_per_flop`: Bits of structure per FLOP
- `loss_per_parameter`: Learning signal per parameter

## Usage

### Training
```python
from toroidal_fractal_intelligence import create_model, train
from toroidal_fractal_intelligence.io.tokenizer import ToroidalTokenizer

model = create_model(d_model=256, n_modes=256)
tokenizer = ToroidalTokenizer("gpt2")

result = train(
    model=model,
    dataset_name="wikitext",
    max_steps=10000,
    learning_rate=3e-4
)
```

### Chat
```python
from toroidal_fractal_intelligence import chat

response = chat(model, tokenizer, "Once upon a time")
print(response)
```

### Interactive Mode
```bash
python -m toroidal_fractal_intelligence.main --mode interactive
```

## Experimental Parameters

Parameters to explore:
- `dimension de l'état` (d_model): 128, 256, 512
- `nombre de phases` (n_modes): 128, 256, 512
- `force de couplage` (coupling_scale): 0.5, 1.0, 2.0
- `pas temporel` (dt): 0.01, 0.1, 0.5
- `mécanisme d'agrégation` (phase_coherence_threshold): 0.5, 0.7, 0.9
- `seuil de consolidation`: 0.5, 0.7, 0.9
- `profondeur hiérarchique`: 3, 5, 7

## Next Steps

1. Implement GPU acceleration for RK4 dynamics
2. Add visualization tools for atom state
3. Experiment with different interaction kernels
4. Test on larger datasets (Common Crawl, Wikipedia)
5. Implement multi-agent collaboration
