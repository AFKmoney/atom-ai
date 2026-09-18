# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 15:37:30 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_obligatory_hard/atom_native.pt`
- Bytes: 27677285
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 16, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "merge_count_total": 0}`
- Training step: `3275000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.004311 | 0.029649 | +0.025338 | 3 |
| 'Qui es-tu ?' | 0.006657 | 0.031680 | +0.025022 | 7 |
| 'Il était une fois' | 0.006141 | 0.030857 | +0.024717 | 6 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.004311 | 0.029649 | +0.025338 | 3 |

- Mean RMS after prompt: **0.005355**
- Mean RMS after 20 gens: **0.030459**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.839724**
- Output state: **0.850928**
- Surface logits: **0.914219**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.9999998669112042,
    0.8722861413880194,
    0.9175094143641228,
    0.9999998669112042
  ],
  [
    0.8722861413880194,
    1.0000000660255866,
    0.9057200819818491,
    0.8722861413880194
  ],
  [
    0.9175094143641228,
    0.9057200819818491,
    0.999999950672156,
    0.9175094143641228
  ],
  [
    0.9999998669112042,
    0.8722861413880194,
    0.9175094143641228,
    0.9999998669112042
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    0.9999998479997035,
    0.7651308840585147,
    0.8541017971057829,
    0.9999998479997035
  ],
  [
    0.7651308840585147,
    1.0000000648823384,
    0.7998768722190879,
    0.7651308840585147
  ],
  [
    0.8541017971057829,
    0.7998768722190879,
    1.0000001033250243,
    0.8541017971057829
  ],
  [
    0.9999998479997035,
    0.7651308840585147,
    0.8541017971057829,
    0.9999998479997035
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.207442**
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
  "train_mse": 5.432273610495031e-05,
  "baseline_mean_mse": 0.0012413040967658162,
  "mse_improvement_vs_mean": 0.001186981360660866,
  "pred_target_cosine_per_prompt": [
    0.9991234120403246,
    0.9976275224308769,
    0.9958653508963412,
    0.9991234120403246
  ],
  "mean_pred_target_cosine": 0.9979349243519668,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000186264528,
  "per_prompt": [
    1.0000000596046494,
    0.999999985098837,
    0.9999999701976758,
    1.0000000596046494
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

