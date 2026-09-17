# Field Intelligence — continuum over parameter count

_Updated: 2026-09-17 (PT)_

## Philosophy

ATOM does not chase GPT-3 by stacking billions of weights. Intelligence here is
**structured persistence in a living field** α: atoms carry phase/energy memory,
MERGE densifies coherent structure, and the surface must **read** α — not ignore
it while CE fits byte bigrams.

Θ stays in the ~8MB class. Capacity grows by **organizing the continuum**, not by
renting GPU farms.

## Why CE-only fails

Post readout-fix (`e48fd6a` lineage): migrate-only surface logit cosine across
prompts reached **~0.938** (field was carrying structure). A short CE smoke
fine-tune pulled inter-prompt logit cosine back toward **~0.99** — the surface
re-learned to ignore α while CE dropped on next-packet bytes.

CE alone rewards *any* map `packet_t → bytes_{t+1}`. A bias-heavy byte head can
satisfy CE while the field is ornamental. Field-aware auxiliaries make ignoring
α *expensive*.

## Field-aware loss

Wired into `AtomNativeModel.transition_loss` (CLI: `--field-loss-weight`,
`--field-contrast-weight`):

1. **Contrastive / margin (α dependence)**  
   Surface logits from true α must differ from logits under **zeroed** and
   **mode-shuffled** α. Hinge on cosine:  
   `relu(cos(true, null) − margin)`.  
   CE still owns byte learning; contrast weight stays small (default `0.1`).

2. **Persistence probe**  
   A tiny linear `field_probe` reconstructs current packet features from
   spectral field features `[mean(α)‖std(α)‖persist‖atom.r]`.  
   Signal: the field must retain packet-shaped structure (default weight `0.05`).

Together: CE learns bytes; field loss keeps the readout **conditioned on α**.

## MERGE (capacity without GPUs)

In toroidal aggregation (`AggregationEngine.merge_coherent`): when two atoms are
**phase-coherent** and energetic enough, replace them with one **heavier** atom
(energy-weighted r/φ/ω/E/κ/M/τ/ρ; ρ+=1). Constituents removed. Merge count logged.

CLI: `--enable-merge` / `--no-enable-merge` (default **on** for new trains).

Structural memory stays **detached** from the current loss graph (same no-grad
atom contribution rule as before). MERGE densifies the atom list so abstraction
sees richer clusters without growing Θ.

## Dual clock

`--slow-every N` (default 1 = always):

- **Every tick**: inject atom → RK4/evolve → surface (and optional MERGE).
- **Every N ticks**, and only if field RMS is relatively stable
  (`|--slow-rms-rel-tol`): aggregation→abstraction + consolidation.

Stable field → safe to consolidate/abstract. Unstable field → keep integrating
the continuum without paying Python cluster cost every step.

## CPU efficiency (profiler hints)

Hot paths observed on box CPU:

| Site | Hint |
|------|------|
| `ToroidalAtomCollection.add` | Lazy dirty-flag buffers — avoid O(n) `stack` per add |
| `AggregationEngine.merge_coherent` | Vectorized coherence + top-k pair scan; cap `max_merges` |
| Stream packet pairs | Use `next(stream_source)` (iterator); ring already deques pending pairs |
| Abstraction/consolidation | Dual clock (`--slow-every`) when RMS stable |
| Atom flush | `--atom-flush-every` bounds collection size in long episodes |

Do **not** rewrite dynamics equations for speed. Profile first (`cProfile` /
`torch.profiler` on `_advance` + `transition_loss`).

## Launch (after stream train ends)

Live stream under `checkpoints/atom_native_stream_dialogue/` must **not** be
written while its process holds the dir. When it finishes, continue with field
intelligence from a **copy** of the best weights:

```bash
mkdir -p checkpoints/atom_native_field_intel
cp checkpoints/atom_native_stream_dialogue/atom_native.pt \
   checkpoints/atom_native_field_intel/atom_native_src.pt

PYTHONPATH=. .venv/bin/python -u tools/run_atom_native.py \
  --stream --data-glob 'data/dialogue_shards/*' --chunk-bytes 1048576 --loop-shards \
  --stream-buffer 2048 \
  --resume checkpoints/atom_native_field_intel/atom_native_src.pt \
  --output-dir checkpoints/atom_native_field_intel \
  --steps 200000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 --episode-length 512 --no-episode-reset \
  --atom-flush-every 256 --field-max-rms 3.0 \
  --learning-rate 2e-4 --surface-learning-rate 6e-4 \
  --energy-decay-min 0.45 --energy-decay-max 0.95 \
  --enable-merge --field-loss-weight 0.05 --field-contrast-weight 0.1 \
  --slow-every 4 --seed 20260917 --log-every 100
```

Probe (surface logit cosine should stay **≪ 0.99**):

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \
  --checkpoint checkpoints/atom_native_field_intel/atom_native.pt
```

## Success bar

- Inter-prompt **surface logit** off-diag cosine **≪ 0.99** after field-aware train
  (migrate-only bar was ~0.938; CE-smoke failure was ~0.99).
- MERGE count > 0 on multi-atom episodes; n_atoms grows slower than ticks.
- Dual clock: slow ticks << total ticks when `--slow-every > 1` and RMS stable.

## Empirical notes (2026-09-17 PT)

| Checkpoint | Surface logit off-diag cos | α off-diag cos | Notes |
|---|---|---|---|
| chat_talk migrate-only | **0.938** | 0.738 | Readout fix success bar |
| readout_fix ~2k CE | ~0.99 | lower | CE re-kills prompt separation |
| field_intel 3k (zero/shuf only, w=0.1) | **0.992** | **0.321** | Train fcos(true,zero)≈0.3–0.5 but inter-prompt still collapses |
| field_intel v2 2.5k (α-bank + w=0.45) | **0.960** | 0.579 | Prompt separation restored below 0.99 bar |

Lesson: contrasting against **zeroed** α is necessary but not sufficient.
CE can learn a shared “non-zero field” logit template. Rolling **α bank**
negatives force surface logits to track *which* field is present.

## How to launch next full train (AFTER stream ends)

Do **not** touch PID holding `checkpoints/atom_native_stream_dialogue/` while it runs.
When stream finishes, copy its weights then continue with field intel:

```bash
mkdir -p checkpoints/atom_native_field_intel
cp checkpoints/atom_native_stream_dialogue/atom_native.pt \
   checkpoints/atom_native_field_intel/atom_native_src.pt

PYTHONPATH=. .venv/bin/python -u tools/run_atom_native.py \
  --stream --data-glob 'data/dialogue_shards/*' --chunk-bytes 1048576 --loop-shards \
  --stream-buffer 2048 \
  --resume checkpoints/atom_native_field_intel/atom_native_src.pt \
  --output-dir checkpoints/atom_native_field_intel \
  --steps 200000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 --episode-length 512 --no-episode-reset \
  --atom-flush-every 256 --field-max-rms 3.0 \
  --learning-rate 1e-4 --surface-learning-rate 3e-4 \
  --energy-decay-min 0.45 --energy-decay-max 0.95 \
  --enable-merge --field-loss-weight 0.08 --field-contrast-weight 0.45 \
  --field-contrast-margin 0.45 --slow-every 4 \
  --seed 20260917 --log-every 100
```

Probe:

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \
  --checkpoint checkpoints/atom_native_field_intel/atom_native.pt
```

Target: surface logit off-diag cosine **≪ 0.99** (v2 smoke hit **0.960**; migrate bar 0.938).
