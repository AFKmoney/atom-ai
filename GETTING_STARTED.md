# Toroidal Fractal Intelligence — Getting Started

> Read [`ATOM_RULES.md`](ATOM_RULES.md) first. The removed historical scripts
> flattened sequences or introduced unrelated baseline architectures.

## What is ATOM?

ATOM (Toroidal Fractal Intelligence) is a novel AI architecture that creates intelligence through **dynamic structures** rather than static weights.

### The Key Idea

Traditional AI:
```
Parameters → Weights → Predictions
```

ATOM:
```
Tokens → Atoms → Structures → Intelligence
```

The system builds **internal structures** from data, and these structures **are** the knowledge.

---

## Quick Start (5 minutes)

### 1. Install

```bash
cd toroidal_fractal_intelligence
pip install -e .
```

### 2. Run the canonical test

```bash
python -m pytest test/core_invariants.py
```

This checks the toroidal projection, RK4 dynamics and finite field behavior.

### 3. Try Chat Mode

```bash
python -m src.main --mode chat --prompt "To be or not to"
```

---

## Understanding the Output

### Training Output

```
Step 100: loss=8.5, atoms=45
Step 200: loss=8.2, atoms=78
Step 300: loss=8.0, atoms=102
```

**What this means:**
- Loss is decreasing (learning)
- Atoms are being created (structure emerging)
- The system is building internal representations

### Chat Output

```
> To be or not to
> ... is the question whether 'tis nobler in the mind to suffer
```

**What this means:**
- The model generates text based on learned structures
- Not pattern-matching, but structure-based generation

---

## Key Concepts

### 1. Toroidal Atom

The basic unit of knowledge:
```python
Atom = {
    'r': position,      # Where in latent space
    'phi': phase,       # Oscillation phase
    'omega': frequency, # Natural frequency
    'E': energy,        # Activation strength
    'kappa': coupling,  # Connection strength
    'M': memory,        # Local memory
    'tau': timescale,   # How fast it changes
    'rho': level        # Hierarchical level
}
```

### 2. Fractal Superposition

Multiple atoms coexist in a shared field:
```python
State = atom_1 + atom_2 + atom_3 + ...
```

The key: **N atoms can be represented efficiently** using spectral methods.

### 3. Continuous Learning

No boundary between training and inference:
```python
while True:
    token = next_token()      # Get new data
    atom = encode(token)      # Convert to structure
    learn(atom)               # Update system
    respond()                 # Generate output
```

---

## Running the canonical training path

```bash
python -m src.main --mode train --dataset wikitext --batch-size 2 --max-steps 2
```

The trainer sends one token per tick and keeps the toroidal state between ticks.
Do not add a flattened benchmark or Transformer baseline; see
[`ATOM_RULES.md`](ATOM_RULES.md).

---

## Interpreting Results

### Key Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| `atoms/parameter` | Structural efficiency | >0.01 |
| `loss improvement` | Learning progress | >10% |
| `structures/FLOP` | Compute efficiency | Higher is better |

### Current Results

```
Tiny Shakespeare:
- 3,718 params → 87 atoms
- Loss: 10.5 → 8.12 (-23%)
- Atoms/parameter: 0.023

Benchmark:
- ATOM: 478K params, 100 atoms, loss=6.22
- Transformer: 41K params, 0 structures, loss=6.29
```

---

## Next Steps

### For Users
1. Run the tests
2. Try chat mode
3. Experiment with different prompts
4. Save and load checkpoints

### For Researchers
1. Read the architecture docs
2. Run the scaling study
3. Compare with baselines
4. Publish results

### For Contributors
1. Fork the repo
2. Add new modules
3. Run tests
4. Submit PR

---

## Troubleshooting

### Problem: NaN loss
**Solution**: Reduce learning rate, check data quality

### Problem: No atoms created
**Solution**: Check encoder output, verify data pipeline

### Problem: Slow training
**Solution**: Reduce batch size, use GPU, decrease seq_len

---

## Resources

- [Full Documentation](DOCUMENTATION.md)
- [Architecture Details](ARCHITECTURE.md)
- [API Reference](USAGE.md)
- [Research Papers](https://arxiv.org/search/?query=toroidal+neural+networks)

---

**Welcome to the future of AI architecture!**
