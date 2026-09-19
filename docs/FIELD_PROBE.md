# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 18:05:33 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_chat/atom_native.pt`
- Bytes: 54533549
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 32, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "merge_coherence_threshold": 0.45, "merge_energy_floor": 0.08, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "printable_aux_weight": 0.0, "field_next_packet_weight": 0.05, "merge_count_total": 72771}`
- Training step: `3498000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.002555 | 0.021276 | +0.018721 | 1 |
| 'Qui es-tu ?' | 0.004479 | 0.022017 | +0.017538 | 1 |
| 'Il était une fois' | 0.003504 | 0.021159 | +0.017655 | 1 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.002555 | 0.021276 | +0.018721 | 1 |

- Mean RMS after prompt: **0.003273**
- Mean RMS after 20 gens: **0.021432**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.615816**
- Output state: **0.845381**
- Surface logits: **0.998685**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.9999999275888765,
    0.9983408628178381,
    0.9984243081442599,
    0.9999999275888765
  ],
  [
    0.9983408628178381,
    0.9999998733360834,
    0.9985799400534325,
    0.9983408628178381
  ],
  [
    0.9984243081442599,
    0.9985799400534325,
    0.9999999216576336,
    0.9984243081442599
  ],
  [
    0.9999999275888765,
    0.9983408628178381,
    0.9984243081442599,
    0.9999999275888765
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    1.0000001198848563,
    0.5723997626100198,
    0.4782710344315834,
    1.0000001198848563
  ],
  [
    0.5723997626100198,
    1.0000000313732404,
    0.5935560518159031,
    0.5723997626100198
  ],
  [
    0.4782710344315834,
    0.5935560518159031,
    0.999999913243217,
    0.4782710344315834
  ],
  [
    1.0000001198848563,
    0.5723997626100198,
    0.4782710344315834,
    1.0000001198848563
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.186113**
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
  "train_mse": 3.6532430385705084e-05,
  "baseline_mean_mse": 0.000621866318397224,
  "mse_improvement_vs_mean": 0.0005853338880115189,
  "pred_target_cosine_per_prompt": [
    0.9996123994309712,
    0.9977090132625108,
    0.9978436500913965,
    0.9996123994309712
  ],
  "mean_pred_target_cosine": 0.9986943655539624,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 0.7915157649440582,
  "per_prompt": [
    0.7680577955314842,
    0.8786917924880981,
    0.7512556762251661,
    0.7680577955314842
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

**field differs across prompts but surface logits nearly identical — field not reliably read by surface (Transformer-regime risk / surface-only byte-LM behavior at decode)**

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

