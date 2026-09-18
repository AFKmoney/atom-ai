# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-17 16:42:00 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_lign_smoke/atom_native.pt`
- Bytes: 20532400
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 16, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.45, 0.95], "enable_merge": true, "slow_every": 4, "field_loss_weight": 0.08, "field_contrast_weight": 0.45, "field_contrast_margin": 0.45, "field_ignorance_weight": 0.08, "field_ignorance_margin": 0.85, "merge_count_total": 0}`
- Training step: `2105000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.003866 | 0.032612 | +0.028746 | 3 |
| 'Qui es-tu ?' | 0.006797 | 0.034235 | +0.027438 | 7 |
| 'Il était une fois' | 0.006036 | 0.033740 | +0.027704 | 6 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.003866 | 0.032612 | +0.028746 | 3 |

- Mean RMS after prompt: **0.005141**
- Mean RMS after 20 gens: **0.033300**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.698694**
- Output state: **0.982931**
- Surface logits: **0.991285**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.999999956863693,
    0.9844636315849673,
    0.9956222308491203,
    0.999999956863693
  ],
  [
    0.9844636315849673,
    1.0000003727045395,
    0.9875383353338196,
    0.9844636315849673
  ],
  [
    0.9956222308491203,
    0.9875383353338196,
    1.0000000619036724,
    0.9956222308491203
  ],
  [
    0.999999956863693,
    0.9844636315849673,
    0.9956222308491203,
    0.999999956863693
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    1.0000002700269885,
    0.47700414684046116,
    0.8353664666956325,
    1.0000002700269885
  ],
  [
    0.47700414684046116,
    0.9999999644925277,
    0.5674224384137968,
    0.47700414684046116
  ],
  [
    0.8353664666956325,
    0.5674224384137968,
    0.9999999362448492,
    0.8353664666956325
  ],
  [
    1.0000002700269885,
    0.47700414684046116,
    0.8353664666956325,
    1.0000002700269885
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.267299**
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
  "train_mse": 7.1824834968481355e-09,
  "baseline_mean_mse": 0.0012413040967658162,
  "mse_improvement_vs_mean": 0.0012412969142823194,
  "pred_target_cosine_per_prompt": [
    1.0000000845191621,
    0.9999991857830204,
    0.9999996325700593,
    1.0000000845191621
  ],
  "mean_pred_target_cosine": 0.999999746847851,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000384946675,
  "per_prompt": [
    1.00000003973643,
    1.0000000447034871,
    1.0000000298023224,
    1.00000003973643
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

