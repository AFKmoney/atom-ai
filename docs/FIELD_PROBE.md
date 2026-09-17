# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-17 11:40:09 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/repos/toroidal-fractal-intelligence/checkpoints/atom_native_chat_talk/atom_native.pt`
- Bytes: 7695498
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 16, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.45, 0.95]}`
- Training step: `2100000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.005059 | 0.049436 | +0.044377 | 3 |
| 'Qui es-tu ?' | 0.007600 | 0.045298 | +0.037698 | 7 |
| 'Il était une fois' | 0.007744 | 0.050234 | +0.042490 | 6 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.005059 | 0.049436 | +0.044377 | 3 |

- Mean RMS after prompt: **0.006366**
- Mean RMS after 20 gens: **0.048601**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.738060**
- Output state: **0.737908**
- Surface logits: **0.997991**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.9999989811540508,
    0.997420346846379,
    0.999363752576907,
    0.9999989811540508
  ],
  [
    0.997420346846379,
    1.0000003715767403,
    0.9943789872243832,
    0.997420346846379
  ],
  [
    0.999363752576907,
    0.9943789872243832,
    0.9999981031995653,
    0.999363752576907
  ],
  [
    0.9999989811540508,
    0.997420346846379,
    0.999363752576907,
    0.9999989811540508
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    0.999999975229589,
    0.6706318291779579,
    0.7363234072234219,
    0.999999975229589
  ],
  [
    0.6706318291779579,
    1.0000000961980724,
    0.6144523478159545,
    0.6706318291779579
  ],
  [
    0.7363234072234219,
    0.6144523478159545,
    0.9999999408803348,
    0.7363234072234219
  ],
  [
    0.999999975229589,
    0.6706318291779579,
    0.7363234072234219,
    0.999999975229589
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.304768**
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
  "train_mse": 1.4885255950503051e-05,
  "baseline_mean_mse": 0.0012413040967658162,
  "mse_improvement_vs_mean": 0.0012264188408153132,
  "pred_target_cosine_per_prompt": [
    0.9997024255365227,
    0.9991344201209608,
    0.9992628284224832,
    0.9997024255365227
  ],
  "mean_pred_target_cosine": 0.9994505249041223,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000347693763,
  "per_prompt": [
    1.0000000397364275,
    1.000000029802325,
    1.000000029802325,
    1.0000000397364275
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

