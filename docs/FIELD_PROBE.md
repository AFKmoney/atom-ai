# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 18:26:59 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_alpha_local/atom_native.pt`
- Bytes: 54312621
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 32, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "merge_coherence_threshold": 0.45, "merge_energy_floor": 0.08, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "printable_aux_weight": 0.0, "field_next_packet_weight": 0.05, "merge_count_total": 82770}`
- Training step: `3508000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.002558 | 0.021940 | +0.019383 | 2 |
| 'Qui es-tu ?' | 0.004515 | 0.024767 | +0.020252 | 3 |
| 'Il était une fois' | 0.003287 | 0.022245 | +0.018958 | 3 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.002558 | 0.021940 | +0.019383 | 2 |

- Mean RMS after prompt: **0.003229**
- Mean RMS after 20 gens: **0.022723**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.709268**
- Output state: **0.855612**
- Surface logits: **0.998711**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.9999997103395062,
    0.9978787600132831,
    0.998678603612428,
    0.9999997103395062
  ],
  [
    0.9978787600132831,
    1.0000000250201064,
    0.9991523031932591,
    0.9978787600132831
  ],
  [
    0.998678603612428,
    0.9991523031932591,
    0.999999959911901,
    0.998678603612428
  ],
  [
    0.9999997103395062,
    0.9978787600132831,
    0.998678603612428,
    0.9999997103395062
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    0.9999999614783628,
    0.6195341075707105,
    0.7037869739411589,
    0.9999999614783628
  ],
  [
    0.6195341075707105,
    1.0000001276887083,
    0.6089658995481689,
    0.6195341075707105
  ],
  [
    0.7037869739411589,
    0.6089658995481689,
    0.9999999643230757,
    0.7037869739411589
  ],
  [
    0.9999999614783628,
    0.6195341075707105,
    0.7037869739411589,
    0.9999999614783628
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.164480**
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
  "train_mse": 5.657849396811798e-05,
  "baseline_mean_mse": 0.000621866318397224,
  "mse_improvement_vs_mean": 0.000565287824429106,
  "pred_target_cosine_per_prompt": [
    0.9992007322619364,
    0.997834047871059,
    0.9958696557603928,
    0.9992007322619364
  ],
  "mean_pred_target_cosine": 0.9980262920388312,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 0.9999999801317851,
  "per_prompt": [
    0.9999999701976776,
    0.9999999403953552,
    1.00000003973643,
    0.9999999701976776
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
- Priming path matches chat: Atomizer encode → `forward_packet(..., merge_enabled=False)` per packet.
- Generation: 20 deterministic packets after prompt (temperature 0.7 path with deterministic=True).
- Did not stop or modify any running training process.

