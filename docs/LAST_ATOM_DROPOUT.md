# last_atom dropout — 2026-09-28

English. Numbers. **No fluency claim.**

## Problem

Under `field_obligatory_hard` + `--last-atom-readout` the byte-tick hard logits are

```
logits = obl·α_MLP(a_hat) + 0.3·frozen_JL(a_hat) + payload + S·last_atom(last_byte, prev_byte, φ)
```

Audit: teacher-forced CE/val is carried by the `last_atom` term (bigram + phase
shortcut); the field (α) path contributes little. Measured here on the b4828 start tip
(4868k): val CE **0.401** normal vs **6.12** with `last_atom` zeroed at eval — worse than
uniform (ln 256 = 5.55). The field path alone is not even a unigram model. In free-run
chat the model falls into attractors (`Bonne`, `Oui,,`, `Peut--`).

Already falsified: fixed/adaptive `last_atom_scale` S (letter soup), `field_hard_couple`
S=0.25/1.0 (W barely trains), free-run-aux (brief diversity, then collapse).

## Lever (opt-in, default OFF)

```
--last-atom-dropout P      # default 0.0 = exact current behavior (no RNG draw)
```

Training only (`module.training` **and** grad enabled). Each surface forward is one
byte-tick position. With probability P that position's `la_byte` term is set to 0
(per-position Bernoulli; no inverted-dropout rescale). On those positions CE has to be
carried by α-MLP + frozen JL. The flag is ignored in eval, generation, validation, and
no-grad predictions. So efference's own-byte pick still uses the full model, the same way
generation does.

* `src/atom_native.py`: `AtomSurfaceHead.last_atom_dropout` (saved in checkpoint config, restored on
  load as provenance). Also adds runtime-only `last_atom_force_off` (not saved), a diagnostic
  that zeroes `la_byte` in any mode.
* `tools/run_atom_native.py`: `--last-atom-dropout`. CLI wins after `--resume`, using the same
  pattern as `field_loss_weight` and `last_atom_scale`. It is recorded in `atomization_stats` / `run_metrics`.
* `test/test_last_atom_dropout.py`:
  - P=0 gives bit-identical logits and draws no global RNG.
  - P=1 in train mode gives exactly the no-last_atom logits.
  - Eval and no-grad are unaffected.
  - P=0.5 is a per-position Bernoulli (~50%).
  - `force_off` works.
  - The value survives a checkpoint roundtrip.
* Diagnostics: `tools/eval_last_atom_ablation.py` rebuilds the exact stream-ring val of
  `run_atom_native` and reports CE normal vs `last_atom`-zeroed. `tools/chat_seeds_probe.py`
  runs the coherence-probe prompts/params with fixed torch seeds.

## Probe (+8k, P=0.5)

Resume: `atom_native_step_4868000_d32_ms1M_lrd_b4828.pt`. This was the newest b4828 tip at probe start,
2026-09-28 13:45 ET. The recipe was the exact live one from `scripts/train_until_coherent.sh`:
`stream-skip 4372000`, d32, `max-span-bytes 1`, `field-obligatory-hard`, `no-enable-merge`,
`no-payload-copy`, `last-atom-readout`, `efference-every 10`, `atom-flush-every 64`,
`episode-length 8000 --episode-reset`, `field-loss-weight 0.05`, contrast 0, LR 1.5e-5 /
SLR 3.6e-5, seed 20260913, `--allow-parallel-train`. The only addition was **`--last-atom-dropout 0.5`**.
Output: `checkpoints/byte_tick_ladrop/atom_native_step_4876000_d32_ms1M_lrd_b4828_ladrop.pt`
(separate dir, so the live `run_metrics.json` was never touched). Baseline: the live
`atom_native_step_4876000_d32_ms1M_lrd_b4828.pt` (same start, same data, P=0).

### Val (stream ring, 256 transitions — identical window for all three rows)

| ckpt | val loss (normal) | byte CE normal | byte CE **last_atom zeroed** |
|------|------------------:|---------------:|-----------------------------:|
| start 4868k (b4828) | 0.4010 | 0.3990 | **6.120** |
| live 4876k (P=0) | 0.4024 | 0.4003 | **6.195** |
| ladrop 4876k (P=0.5) | 0.5363 | 0.5309 | **3.399** |

Reference on the same window, from corpus counts: unigram CE ≈ 3.24, bigram CE ≈ 1.77, uniform 5.55
(`val_window_baselines.json`). The eval script reproduces `run_metrics` val exactly (0.4010 /
0.4024 / 0.5363).

Read it plainly:
* The field path alone went from worse-than-uniform (6.12) to about unigram level (3.40) in 8k
  steps. It learned the byte marginal. It did not learn context: 3.40 is still far above the
  bigram's 1.77.
* Normal val got worse, 0.401 → 0.536 (+0.13), because `last_atom` now only has half the
  positions to fit.
* The live P=0 run moved field-alone CE the other way (6.12 → 6.20). Without pressure, the field path decays.

### Chat

The coherence probe CLI takes only the **first line** of the response (`sed … | head -1`), with
unseeded sampling, T=0.8, top-k 12, `--no-role-prime`, 96 packets, max-length 100:

| prompt | live 4876k | ladrop 4876k |
|--------|-----------|--------------|
| Bonjour, comment ça va ? | `Bonne` | `Bonne` |
| Qui es-tu ? | `Oui,,` | `Bonne` |
| Il était une fois | `Oui,,` | `Bonne` |

Seeded probe (seeds 1,2,3 × T 0.6/0.8/1.0 × 3 prompts = 27 full replies per ckpt,
`chat_seeds_*.txt`):

| ckpt | unique replies /27 | mean distinct replies across the 3 prompts (same T, seed) | first words |
|------|------------------:|-----------------------:|-------------|
| start 4868k | 23 | 2.56 | Oui,, 9 / Je 9 / Peut-- … |
| live 4876k | 21 | 2.56 | Je 16 / Peut-- … |
| ladrop 4876k | 15 | **1.67** | **Je 27/27** |

Example, T=0.8 seed=1, same reply for all three prompts on ladrop:
`'Je …\nsoirends …\nAsssistant:::::::::::::::::::O   \nuurrraais'`.
Chat got **worse**. Replies depend less on the prompt and always open with `Je`, which fits a field
path that learned only the unconditional marginal and now biases every opening.

### Other observations

* Train-log `field=96.0 rms=3.0 scale≈0.999` on both runs: α sits pinned at the
  `field_max_rms=3.0` clamp for most of each episode. A saturated field can carry little
  position-specific information. This is a plausible reason the field path tops out at unigram.
* Median train loss: live 0.039, ladrop 2.98 (half the positions have no shortcut).

## Verdict

P=0.5 for 8k steps is a **technical PASS, not a fluency lever yet**. It proves the field path
can be pushed to carry probability mass (field-alone CE 6.12 → 3.40), but what it learned
is the marginal, not context. Chat is less prompt-conditioned. Normal val is +0.13 worse.

## Next (not done here)

1. Longer P=0.5 run (≥40k) tracking **field-alone CE**. If it stays near unigram (~3.2), the
   α state holds no usable context, and the bottleneck is upstream (saturated field at the RMS
   clamp, α features), not the readout.
2. P=1.0 for 8k steps is a clean "field-only" ceiling test. It matches P=0.5 on field-alone CE if the field
   is capacity-limited. Do not chat-evaluate it with `last_atom` on (train/eval mismatch).
3. Only if field-alone CE clearly beats bigram (≲1.8): consider a stacked recipe (P anneal
   0.5→0.2).

## Artifacts

`docs/artifacts/last_atom_dropout/`: `val_ablation.{json,txt}`, `val_window_baselines.json`,
`chat_seeds_{start_4868000,live_4876000,ladrop_4876000}.{json,txt}`,
`gen_coherent_4876000_{b4828,ladrop}.txt`, `run_metrics_*`, `train_ladrop.log`,
`yield_to_live.py` + `measure.sh` (see CPU note).

## CPU note (parallel probes)

The live trainer runs torch with 8 OMP threads that busy-wait. Any other busy process on
one core cut live throughput from about 77 to about 2 steps/s (measured with SIGSTOP on and off). The probe
and evals were therefore run single-threaded (`OMP_NUM_THREADS=1`) under
`yield_to_live.py`, which SIGSTOPs only the probe's own process group while the live chunk is past its
single-threaded stream-skip phase. The live loop was never signalled.
