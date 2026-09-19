# FIELD_PROBE — persistence vs surface-only byte-LM

_Generated: 2026-09-18 17:04:41 PDT_

## Thesis

We still train ATOM like a small LM: `x[t] → model → packet x[t+1]` CE.
The field can be ignored while the surface learns byte bigrams.
This probe judges the **field** (alpha / consolidation / atoms), not only CE/gen.

## Checkpoint

- Path: `/workspace/exports/atom-ai/checkpoints/atom_native_obligatory_hard/atom_native.pt`
- Bytes: 47380837
- Config: `{"d_model": 64, "n_modes": 64, "n_atoms_max": 512, "max_payload_bytes": 32, "atomizer_version": "atomizer-v1-byte-span", "field_max_rms": 3.0, "energy_decay_bounds": [0.3, 0.95], "enable_merge": true, "slow_every": 1, "field_loss_weight": 0.05, "field_contrast_weight": 0.1, "field_contrast_margin": 0.55, "field_ignorance_weight": 0.0, "field_ignorance_margin": 0.85, "field_ignorance_ablate_shared": false, "field_ignorance_prompt_bank": false, "field_obligatory_readout": true, "field_obligatory_hard": true, "printable_aux_weight": 0.0, "merge_count_total": 0}`
- Training step: `3425000`

## Hard numbers

| Prompt | RMS after prompt | RMS after 20 gen | Δ RMS | n_atoms after prompt |
|--------|------------------|------------------|-------|----------------------|
| 'Bonjour' | 0.002521 | 0.022426 | +0.019905 | 2 |
| 'Qui es-tu ?' | 0.004604 | 0.022737 | +0.018133 | 3 |
| 'Il était une fois' | 0.003697 | 0.023243 | +0.019546 | 3 |
| 'Utilisateur: Bonjour\nAssistant:' | 0.002521 | 0.022426 | +0.019905 | 2 |

- Mean RMS after prompt: **0.003336**
- Mean RMS after 20 gens: **0.022708**

### Prompt sensitivity (off-diagonal mean cosine)

- Alpha (field): **0.713154**
- Output state: **0.863222**
- Surface logits: **0.866300**
- Persistence: **nan**

Interpretation: cosine ≈ 1.0 ⇒ surface/field ignore prompt differences; meaningfully < 1 ⇒ prompt-sensitive state.

### Pairwise surface-logit cosine matrix

```
[
  [
    0.9999998555154952,
    0.8297065604746837,
    0.8437657705714171,
    0.9999998555154952
  ],
  [
    0.8297065604746837,
    0.9999998029154439,
    0.8508563981215139,
    0.8297065604746837
  ],
  [
    0.8437657705714171,
    0.8508563981215139,
    1.0000000147683163,
    0.8437657705714171
  ],
  [
    0.9999998555154952,
    0.8297065604746837,
    0.8437657705714171,
    0.9999998555154952
  ]
]
```

### Pairwise alpha cosine matrix

```
[
  [
    0.9999996713840237,
    0.6621227539248857,
    0.6435238808772797,
    0.9999996713840237
  ],
  [
    0.6621227539248857,
    0.9999997503523191,
    0.6676299859751356,
    0.6621227539248857
  ],
  [
    0.6435238808772797,
    0.6676299859751356,
    0.9999999524523814,
    0.6435238808772797
  ],
  [
    0.9999996713840237,
    0.6621227539248857,
    0.6435238808772797,
    0.9999996713840237
  ]
]
```

## Field reconstruction probe

- Field differs across prompts: **True**
- Persistence differs: **False**
- Mean pairwise alpha L2: **0.172062**
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
  "train_mse": 4.610618998412974e-05,
  "baseline_mean_mse": 0.000621866318397224,
  "mse_improvement_vs_mean": 0.0005757601284130942,
  "pred_target_cosine_per_prompt": [
    0.9994798849544735,
    0.99740065679724,
    0.9970745391964889,
    0.9994798849544735
  ],
  "mean_pred_target_cosine": 0.9983587414756689,
  "note": "In-sample fit only (P prompts). Success means field features linearly span last-k packet features on this set \u2014 not a general decoder."
}
```

Atom collection probe:
```json
{
  "status": "ok",
  "mean_last_k_compiled_vs_stored_r_cosine": 1.0000000149011619,
  "per_prompt": [
    1.0000000298023242,
    0.999999960263569,
    1.00000003973643,
    1.0000000298023242
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

