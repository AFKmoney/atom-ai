# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 17:15:40 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_obligatory_hard/atom_native.pt`
- Bytes: 46757429
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 32, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "merge_coherence_threshold": 0.45, "merge_energy_floor": 0.08, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "printable_aux_weight": 0.0, "merge_count_total": 7968}`
- Training step: `3433000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.002505 | 0.020182 | +0.017677 | 1 |
| 'Qui es-tu ?' | 0.004638 | 0.023975 | +0.019337 | 1 |
| 'Il était une fois' | 0.003720 | 0.024232 | +0.020513 | 1 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.002505 | 0.020182 | +0.017677 | 1 |

- Mean RMS after prompt: **0.003342**
- Mean RMS after 20 gens: **0.022143**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.704931**
- Output state: **0.854482**
- Surface logits: **0.859962**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    1.0000002032292556,
    0.8166799107515312,
    0.8341128545816225,
    1.0000002032292556
  ],
  [
    0.8166799107515312,
    1.0000001357550545,
    0.8581846253325931,
    0.8166799107515312
  ],
  [
    0.8341128545816225,
    0.8581846253325931,
    1.000000369780937,
    0.8341128545816225
  ],
  [
    1.0000002032292556,
    0.8166799107515312,
    0.8341128545816225,
    1.0000002032292556
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    1.0000002836786583,
    0.6407488731245381,
    0.6348513743873437,
    1.0000002836786583
  ],
  [
    0.6407488731245381,
    1.0000001432707264,
    0.6783866577036988,
    0.6407488731245381
  ],
  [
    0.6348513743873437,
    0.6783866577036988,
    1.0000000031143206,
    0.6348513743873437
  ],
  [
    1.0000002836786583,
    0.6407488731245381,
    0.6348513743873437,
    1.0000002836786583
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.174867**
- Alpha re-prime 1-NN accuracy: **0.750**
- Reconstruction succeeds (diagnostic): **True**
- Summary: partial/diagnostic linear map from field features to last-k packet features works in-sample

Linear probe:
```json
{
  "status": "ok",
  "method": "dual_ridge_least_squares",
  "lambda": 0.01,
  "n_prompts": 4,
  "field_dim": 4224,
  "feature_dim": 320,
  "train_mse": 4.497965710470453e-05,
  "baseline_mean_mse": 0.000621866318397224,
  "mse_improvement_vs_mean": 0.0005768866612925194,
  "pred_target_cosine_per_prompt": [
    0.9995301254190327,
    0.9974805077514372,
    0.9970189133176329,
    0.9995301254190327
  ],
  "mean_pred_target_cosine": 0.9983899179767839,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 0.7887037515389569,
  "per_prompt": [
    0.7596256732940674,
    0.8991588233898667,
    0.7364048361778259,
    0.7596256732940674
  ],
  "note": "Atom collection stores detached compiled atoms from this episode; high cosine is expected (same episode write). This is NOT recovery from field alone \u2014 it shows structural memory buffer retention."
}
```

### Not possible yet

- MERGE of packets into durable field structures
- SPLIT / dual-clock span control
- Dedicated field → past-packet decoder
- Retrieving arbitrary past payloads from alpha alone as text

## Verdict

**field is carrying structure (prompt-sensitive; diagnostic reconstruction partial)**

CE down + gen still noise without field reconstruction = still Transformer regime without the farm.

## Next levers (not implemented in this probe)

- Real **MERGE** (packets → durable field structures)
- **Dual clocks** (fast surface / slow field)
- **Adaptive spans** (beyond fixed max_span_bytes)

## Method notes

- Read-only: `model.eval()`, all `requires_grad=False`, no optimizer.
- Priming path matches chat: Atomizer encode → `forward_packet` per packet.
- Generation: 20 deterministic packets after prompt (temperature 0.7 path with deterministic=True).
- Did not stop or modify any running training process.

