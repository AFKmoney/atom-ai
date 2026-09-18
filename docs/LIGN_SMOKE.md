# L_ign smoke (+2500) — 2026-09-17

English. Numbers + mechanism notes. No fluency claim.

## Setup

| Item | Value |
|------|-------|
| code | `atom-ai` main (≥ `a01e6b4`; train ran under tree that became `81d9444`) |
| resume | `atom_native_src.pt` @ step **2102500** (field_intel) |
| out | `checkpoints/atom_native_lign_smoke/atom_native.pt` @ **2105000** |
| steps | **2500** |
| data | `data/corpus_train_chat.txt` (~778 KB) |
| L_ign | weight **0.08**, margin **0.85** |
| also on | field-loss 0.08, contrast 0.45 / margin 0.45, merge ON, slow_every 4 |
| stream | **not touched** (PID left running) |

Artifacts: `docs/artifacts/lign_smoke_2k5/`.

## Probe (inter-prompt off-diag mean cosine)

| | before | after |
|--|--------|-------|
| surface logits | **0.992294** | **0.991285** |
| α (field) | **0.720524** | **0.698694** |

Criterion was: logits drop vs prior field_intel bar **~0.960**.  
Result: **fail** — logits stay **~0.991** (Δ ≈ −0.001). α not killed (mild −0.022).

## Train log (selected)

```
run 1/2500     fcos=0.999  ign=0.000  mph=0.000  rms=0.002
run 200/2500   fcos=0.205  ign=0.150  mph=0.500  rms=0.155
run 2400/2500  fcos=0.259  ign=0.150  mph=0.500  rms=3.000
run 2500/2500  fcos=0.160  ign=0.150  mph=0.500  rms=3.000
```

- **fcos** (train true-vs-zero style): collapses early, stays ~0.09–0.28.
- **ign**: sticks at **0.150** after warmup ⇒ `ReLU(cos_ign − 0.85) = 0.15` ⇒ **cos_ign ≈ 1.0** every logged step.
- **merges**: **0** all steps.
- **mph** (max phase coherence pre-threshold): **0.500** after warmup (threshold still 0.55 — never merges).
- **field_max_rms=3.0**: saturated much of the second half.

Effective L_ign contribution ≈ `0.08 × 0.15 = 0.012` vs CE — too small to move the shared logit template.

## How ℓ(α) vs ℓ(α_bar) is computed (code path)

Location: `src/atom_native.py` → `AtomNativeModel._field_auxiliary_losses`.

1. `forward_packet` → `output["surface"]["byte_logits"]` flattened as `true_flat` = **ℓ(α)** from the live forward (same surface head as CE).
2. Bank update happens in `transition_loss` **before** aux: append `sg[α]` to `_alpha_bank` (max 8).
3. For L_ign, `others = bank[:-1]` excluding near-duplicates of current α; `α_bar = mean(others).detach()`.
4. Second surface call: `self.surface(None, alpha=α_bar, persistent_state=persist_state, atom_r=atom_r)` → **ℓ(sg[α_bar])**.
5. `L_ign = ReLU( cos(true_flat, ℓ(α_bar)) − m )` with `m=0.85`.

### Same readout path?

**Yes** — identical `AtomSurfaceHead` / `self.surface(...)`.

**Shared non-α inputs on both calls:** `persistent_state` and `atom_r` are **not** swapped with the bank. Only `alpha` changes. If the head leans on atom_r / persistence / “field non-zero” features, ℓ(α) ≈ ℓ(α_bar) even when α differs.

### Bank too close?

Bank holds recent episode fields (same train stream), size 8. Mean of others can stay near current α in a CE-dominated regime → cos_ign→1 → hinge saturates at `1−m=0.15`.

### Margin / weight?

- Margin 0.85 with cos≈1.0 ⇒ constant hinge 0.15 (gradient may be weak / saturated).
- Weight 0.08 ⇒ tiny vs CE + contrast block.

## Verdict

L_ign **as wired** did not break the shared surface template. Field still carries structure (α cosine ~0.70). Next scientific move = **re-inspect readout coupling** (one mechanism), not more CE steps. Candidates (pick **one** later):

1. Ablate shared `atom_r` / `persistent_state` on the α_bar surface call (force α-only readout for the hinge).
2. Bank negatives from **distinct prompts** (probe-style), not adjacent train ticks.
3. Replace saturated cosine hinge with a stronger divergence on logits (still one term).

Do **not** retune MERGE thresholds in the same change. Do **not** claim fluent chat.

## Related

- Formula flags: `docs/FIELD_IGNORANCE.md`
- Probe tables: `docs/FIELD_PROBE.md` (updated to post-smoke ckpt)
- CPU serialize/prefetch: `docs/CPU_PERF.md` (main `81d9444`+)
