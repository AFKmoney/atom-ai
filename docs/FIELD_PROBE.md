# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 16:08:04 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_obligatory_hard/atom_native.pt`
- Bytes: 27365541
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 16, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "printable_aux_weight": 0.08, "merge_count_total": 0}`
- Training step: `3325000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.004300 | 0.029990 | +0.025691 | 3 |
| 'Qui es-tu ?' | 0.006659 | 0.031848 | +0.025189 | 7 |
| 'Il était une fois' | 0.006217 | 0.031357 | +0.025140 | 6 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.004300 | 0.029990 | +0.025691 | 3 |

- Mean RMS after prompt: **0.005369**
- Mean RMS after 20 gens: **0.030796**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.836536**
- Output state: **0.851050**
- Surface logits: **0.912255**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.999999863875923,
    0.8666357052871798,
    0.9235450474693955,
    0.999999863875923
  ],
  [
    0.8666357052871798,
    0.9999999093416008,
    0.8931712837450864,
    0.8666357052871798
  ],
  [
    0.9235450474693955,
    0.8931712837450864,
    1.000000023423159,
    0.9235450474693955
  ],
  [
    0.999999863875923,
    0.8666357052871798,
    0.9235450474693955,
    0.999999863875923
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    1.0000002634879106,
    0.7558299412960835,
    0.8646300217935912,
    1.0000002634879106
  ],
  [
    0.7558299412960835,
    0.9999999476791628,
    0.7782952142356427,
    0.7558299412960835
  ],
  [
    0.8646300217935912,
    0.7782952142356427,
    1.0000001239970564,
    0.8646300217935912
  ],
  [
    1.0000002634879106,
    0.7558299412960835,
    0.8646300217935912,
    1.0000002634879106
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.210736**
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
  "train_mse": 5.320868149283342e-05,
  "baseline_mean_mse": 0.0012413040967658162,
  "mse_improvement_vs_mean": 0.0011880954152729828,
  "pred_target_cosine_per_prompt": [
    0.9990714752923866,
    0.9978818230163662,
    0.9959597249060029,
    0.9990714752923866
  ],
  "mean_pred_target_cosine": 0.9979961246267856,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000571211198,
  "per_prompt": [
    1.0000000397364335,
    1.0000000596046412,
    1.0000000894069707,
    1.0000000397364335
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

