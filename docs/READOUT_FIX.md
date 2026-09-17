# READOUT_FIX — surface decode reads living field α

_Updated: 2026-09-17 12:32:30 PDT_

## Problem

Probe (`tools/probe_field_persistence.py`) on
`checkpoints/atom_native_chat_talk/atom_native.pt` showed:

| Metric | Before |
|--------|--------|
| Mean off-diag cosine **α** (field) | **~0.738** |
| Mean off-diag cosine **output_state** | ~0.738 |
| Mean off-diag cosine **surface logits** | **~0.998** |
| Generation | Identical collapse (`\ne  ` / shared noise) |

The living toroidal field **did** differ across prompts, but
`AtomSurfaceHead` effectively ignored it: decode behaved like a last-token /
bias-dominated byte-LM. Shared decoder bias pinned logits so inter-prompt
cosine stayed ≈ 1.0 even after LayerNorm.

## Fix (readout only — no MERGE)

Architecture change in `src/atom_native.py`:

1. **`AtomSurfaceHead` field-first contract**
   - Spectral features: `[mean(α) ‖ std(α) ‖ consolidation.persistence ‖ atom.r]`
   - `field_to_state`: compress → `d_model` (init: identity on mean, 0.5·I on std)
   - Gated mix with legacy mean-state: `σ(field_gate)·field + (1−σ)·legacy`
   - **Bias-free field→logit skip** (`field_byte_skip` / `field_length_skip`) scaled by
     `softplus(skip_gate)` so α differences reach logits even when `|b|` is large
   - LayerNorm + legacy linears retained for Θ continuity

2. **`_advance` / `_output_state`**
   - Every tick passes **full α**, always-on consolidation buffer, and current
     `atom.r` into the surface (not only `mean(α)` / last packet state)

3. **Backward-compatible load**
   - Old weights load by shape filter
   - Missing field-readout keys → seeded `_init_field_readout()`, bias ×0.05,
     `skip_gate ← 4.0` so migrated checkpoints separate logits **before** retrain
   - Fresh from-scratch init uses softer `skip_gate=1.0` so CE training stays stable

No Transformer / attention / GPT-2 / QKV. Θ stays ~8MB class (~5.8M trainable
params at d_model=64). Stream train under `atom_native_stream_dialogue` was
**not** stopped or modified.

## Before / after probe

Checkpoint: `checkpoints/atom_native_chat_talk/atom_native.pt` (step 2.1M),
read-only eval after migration (no weight update required for separation).

| Metric | Before | After readout fix (migrate load) |
|--------|--------|----------------------------------|
| Mean off-diag cosine surface logits | **0.997991** | **~0.938** |
| Mean off-diag cosine α | 0.738060 | 0.738060 |
| Mean RMS after prompt | ~0.0064 | ~0.0064 |
| Mean RMS after 20 gen | ~0.0486 | ~0.0485 |
| Verdict | field differs, surface unread (byte-LM decode) | **field carrying structure (prompt-sensitive)** |

Sample gens (deterministic, role-primed) are **prompt-distinct** (no longer
identical collapse). Without a surface re-fit they remain embryonic/noisy.

### Optional tiny smoke fine-tune

`checkpoints/atom_native_readout_fix/` — 2k steps from a **copy** of chat_talk
(`corpus_train_chat.txt`), stream dir untouched. After 2k: α cosine fell to
~0.39 (more field separation) but surface logit cosine rose toward ~0.99 as the
skip/gate adapted under CE — longer surface-focused training still needed for
fluent decode. Migration-only numbers above are the readout-fix success bar.

## Tests

- `test/test_generation_smoke.py::FieldReadoutSensitivityTests`
  - Synthetic distinct α → logit cosine < 0.95
  - Migrated chat_talk ckpt → Bonjour vs Qui es-tu logit cosine < 0.95

## Not done (explicit)

- Full **MERGE** / dual clocks / adaptive spans — deferred
- No changes to the live stream trainer PID / `atom_native_stream_dialogue`

## How to re-probe

```bash
PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py \
  --checkpoint checkpoints/atom_native_chat_talk/atom_native.pt
```
