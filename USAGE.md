# Toroidal Fractal Intelligence — Usage Guide

> ATOM is stateful and tick-based. Read [`ATOM_RULES.md`](ATOM_RULES.md) before
> writing a training or evaluation loop.

## Installation

```bash
cd toroidal_fractal_intelligence
pip install -e .
```

## Quick Start

### 1. Train from Scratch

```bash
python -m toroidal_fractal_intelligence.main \
    --mode train \
    --dataset wikitext \
    --max-steps 10000 \
    --batch-size 32 \
    --learning-rate 3e-4
```

### 2. Chat with Trained Model

```bash
python -m toroidal_fractal_intelligence.main \
    --mode chat \
    --checkpoint checkpoints/final_model.pt \
    --prompt "Once upon a time"
```

### 3. Interactive Mode

```bash
python -m toroidal_fractal_intelligence.main \
    --mode interactive \
    --checkpoint checkpoints/final_model.pt
```

## Python API

### Create Model

```python
from toroidal_fractal_intelligence import create_model

model = create_model(
    vocab_size=32000,
    d_model=256,
    n_modes=256,
    n_atoms_max=1024,
    device="cuda"  # or "cpu"
)
```

### Train Model

```python
from toroidal_fractal_intelligence import train

result = train(
    model=model,
    dataset_name="wikitext",
    dataset_subset="wikitext-2-raw-v1",
    batch_size=32,
    max_steps=10000,
    learning_rate=3e-4,
    save_dir="./checkpoints"
)

# The trainer executes token[t] -> logits[t] -> target token[t+1]
# sequentially; never flatten a complete sequence before model(...).
```

### Generate Text

```python
from toroidal_fractal_intelligence import chat
from toroidal_fractal_intelligence.io.tokenizer import ToroidalTokenizer

tokenizer = ToroidalTokenizer("gpt2")
response = chat(model, tokenizer, "The future of AI is")
print(response)
```

### Load Model

```python
model.load("checkpoints/final_model.pt")
```

### Save Model

```python
model.save("my_model.pt")
```

## Configuration

### Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vocab_size` | 32000 | 1000-100000 | Token vocabulary size |
| `d_model` | 256 | 64-1024 | Hidden dimension |
| `n_modes` | 256 | 64-1024 | Spectral modes |
| `n_atoms_max` | 1024 | 128-4096 | Max concurrent atoms |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Samples per batch |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `max_steps` | 10000 | Training iterations |
| `log_interval` | 100 | Steps between logs |
| `save_interval` | 1000 | Steps between saves |

## Understanding the Output

### Training Output

```
Step 100: loss=2.3456, perplexity=10.43, atoms=12, aggregates=3
Step 200: loss=1.8923, perplexity=6.63, atoms=28, aggregates=7
...
```

- `loss`: Cross-entropy loss
- `perplexity`: exp(loss), lower is better
- `atoms`: Current number of toroidal atoms
- `aggregates`: Number of aggregate structures

### Model Status (interactive mode)

```
Model status:
  Atoms: 156
  Modes: 256
  d_model: 256
  Coupling scale: 1.0234
  Energy decay: 0.9901
  Phase sync: 0.4892
```

## Checking Results

### View Training Log

```bash
cat checkpoints/training_log_step_10000.json
```

### View Metrics Summary

```python
from toroidal_fractal_intelligence.evaluation.metrics import ToroidalMetrics

metrics = ToroidalMetrics()
# Metrics are automatically logged during training
summary = metrics.get_summary()
print(summary)
```

## Advanced Usage

### Custom Dataset

```python
from toroidal_fractal_intelligence.io.data import ToroidalTextDataset, create_training_dataloader

# Create custom dataset
dataset = ToroidalTextDataset(
    texts=["Sample text 1...", "Sample text 2..."],
    tokenizer=tokenizer,
    max_length=128
)

dataloader = create_training_dataloader(
    dataset_name="custom",
    batch_size=16,
    infinite=False
)
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup distributed training
dist.init_process_group(backend="nccl")
model = create_model(device=f"cuda:{local_rank}")
model = DDP(model, device_ids=[local_rank])

# Train as normal
```

### Continuous Learning

```python
# Load existing model
model.load("checkpoints/final_model.pt")

# Continue training on new data
result = train(
    model=model,
    dataset_name="new_dataset",
    max_steps=5000
)
# Model continues learning without forgetting
```

## Troubleshooting

### Out of Memory

Reduce `n_atoms_max` and `n_modes`:
```python
model = create_model(d_model=128, n_modes=128, n_atoms_max=512)
```

### Slow Training

Enable GPU:
```python
model = create_model(device="cuda")
```

Or reduce `n_steps` in dynamics:
```python
# Modify dynamics.py: n_steps=2
```

### Poor Performance

Increase training steps:
```bash
--max-steps 50000
```

Or increase model capacity:
```python
model = create_model(d_model=512, n_modes=512)
```
