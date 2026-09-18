# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 14:19:31 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_obligatory_hard/atom_native.pt`
- Bytes: 20564037
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 16, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "merge_count_total": 0}`
- Training step: `3125000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.004295 | 0.033326 | +0.029031 | 3 |
| 'Qui es-tu ?' | 0.006653 | 0.032304 | +0.025651 | 7 |
| 'Il était une fois' | 0.005975 | 0.032279 | +0.026304 | 6 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.004295 | 0.033326 | +0.029031 | 3 |

- Mean RMS after prompt: **0.005304**
- Mean RMS after 20 gens: **0.032809**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.841826**
- Output state: **0.846270**
- Surface logits: **0.915846**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    1.0000000561175475,
    0.878994266604593,
    0.9177184374192179,
    1.0000000561175475
  ],
  [
    0.878994266604593,
    1.0000000655942185,
    0.9016519853899178,
    0.878994266604593
  ],
  [
    0.9177184374192179,
    0.9016519853899178,
    0.9999998010950305,
    0.9177184374192179
  ],
  [
    1.0000000561175475,
    0.878994266604593,
    0.9177184374192179,
    1.0000000561175475
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    0.9999997381606669,
    0.7706877091310306,
    0.8453447297830701,
    0.9999997381606669
  ],
  [
    0.7706877091310306,
    1.0000000017812458,
    0.8188900243908617,
    0.7706877091310306
  ],
  [
    0.8453447297830701,
    0.8188900243908617,
    1.000000033734737,
    0.8453447297830701
  ],
  [
    0.9999997381606669,
    0.7706877091310306,
    0.8453447297830701,
    0.9999997381606669
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.203277**
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
  "train_mse": 5.8431025536265224e-05,
  "baseline_mean_mse": 0.0012413040967658162,
  "mse_improvement_vs_mean": 0.001182873071229551,
  "pred_target_cosine_per_prompt": [
    0.9991273961043431,
    0.9973810043138795,
    0.9954038251210716,
    0.9991273961043431
  ],
  "mean_pred_target_cosine": 0.9977599054109093,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000447034858,
  "per_prompt": [
    1.0000000596046459,
    1.000000014901162,
    1.000000044703489,
    1.0000000596046459
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

