# Field saturation — `--field-leak` (2026-09-28)

English. Numbers. **No fluency claim.**

## Diagnosis (4876k live + ladrop, eval/no_grad, 8k-tick episode)

The RMS clamp is `FieldAmplitudeController` in `src/atom_native.py`: after each
tick's RK4 evolve + phase-sync, if `rms(α) > field_max_rms` (CLI default **3.0**)
then `α ← α · (cap / rms)`. Directions are kept; amplitude is hard-capped.

What drives the field to the cap:

| knob | live / ladrop 4876k value | effect |
|------|--------------------------:|--------|
| `energy_decay` | **0.300** (floor of `[0.3, 0.95]`) | linear rate ≈ `sin(φ_m) + (ed−1)`; modes with `sin(φ_m) > 0.7` (m ≈ 4…12) are **unstable** and grow every tick |
| `coupling_scale` (drive gain) | ≈ 0.87 | multiplies the mode-modulated token drive |
| `phase_sync` | ≈ 0.07 | weak neighbour coupling |
| `dt`, RK4 steps | 0.01 × 4 | small step; growth is from the unstable modes, not the integrator |
| persistence / consolidation | EMA 0.995 | slow buffer; not the cause of the RMS pin |
| efference | every 10 | commits predicted bytes into the atom list; does not change the α integrator |

Episode numbers (corpus offset 123457, atom-flush every 64, same for both ckpts):

| | live 4876k | ladrop 4876k |
|--|-----------:|-------------:|
| first tick at cap | **3036** | **2937** |
| fraction of ticks at cap | **0.621** | **0.633** |
| after first hit, still at cap | **1.0** | **1.0** |
| temporal cos(α_t, α_{t+k}) for k=1…256 (2nd half) | **1.000** | **1.000** |
| inter-context α cosine after 64-byte forks @P=256 | 0.992 | 0.995 |
| same @P=3000 (at cap) | **1.000** | **1.000** |
| linear probe next-byte CE from `a_hat` (JL) | — | 3.19 ≈ unigram 3.21 |

Injection itself is text-dependent (`atom.r` cos-to-mean ≈ 0.5–0.7, var-fraction
≈ 0.5–0.8). The pin is **not** "the drive ignores text"; it is "unstable modes
integrate until the hard clamp, then every text lands on the same ray."

Frozen-weight λ sweep on ladrop 4876k (eval-only `--field-leak`; no retrain):

| λ | frac@cap (4k-tick ep.) | rms@4k | fork α-cos @P=3000 | next-byte probe CE | cur-byte CE |
|--:|---------:|-------:|-------------------:|-------------------:|------------:|
| 0 | 0.27 | 3.00 | 1.000 | 3.19 | 3.18 |
| 0.01 | 0 | 0.047 | 0.979 | 2.99 | 2.75 |
| 0.03 | 0 | 0.020 | 0.937 | 2.78 | 2.35 |
| 0.10 | 0 | 0.0065 | 0.886 | 2.45 | 1.52 |
| **0.30** | **0** | **0.0026** | **0.794** | **2.08** | **0.46** |

λ = 0.3 is the strongest "off the cap + text-dependent field" point on that
sweep without inventing a new parameter family. That is the probe value.

## Lever (opt-in, default OFF)

```
--field-leak λ      # default 0.0 = exact current behavior (no-op)
```

Each `_advance` tick, **before** `add_atom_contribution`:

```
if λ > 0:  α ← (1 − λ) · α     # no_grad; α Parameter is state, not a weight
```

Then the rest of the tick is unchanged (inject → RK4 → phase-sync → RMS clamp →
surface). λ = 0 takes the **exact** branch it always took (the `if` is skipped),
so the live recipe with no flag is bit-identical to pre-lever HEAD.

Why this and not the alternatives:

* **Higher `field_max_rms`** only delays the pin; unstable modes still dominate.
* **Soft renorm** (always project onto the sphere) forces every tick onto the
  cap surface — the saturated regime we are trying to leave.
* **Scaled / learnable drive gain** fights the floor on `energy_decay`; the
  unstable modes remain until decay is allowed above ~0.7, which is a different
  (and riskier) change to always-on dynamics repair.
* **Input-dependent decay** needs a new learned head. Leak is one scalar, applied
  only when opted in, and turns α into a contractive recency-weighted memory
  whose horizon is ≈ 1/λ ticks (λ = 0.3 → ~3 ticks; the surface still sees the
  full living-atom list + last_atom bigram).

* `src/atom_native.py`: `AtomNativeModel.field_leak` (saved in checkpoint config,
  restored on load so chat/eval see the trained dynamics). Absent key = 0.0.
* `tools/run_atom_native.py`: `--field-leak`. CLI wins after `--resume` (same
  pattern as `last_atom_dropout` / `field_loss_weight`). Recorded in
  `atomization_stats` / `run_metrics`.
* `test/test_field_leak.py`: bit-identical to commit `90f7462` at λ=0 (full
  train+eval trace); λ is exactly pre-injection scaling; λ keeps α off a tight
  cap where λ=0 saturates; checkpoint roundtrip + old-config→0; CLI default and
  CLI-wins-after-resume.
* Diagnostic: `tools/diag_field_saturation.py`.

## Probe (+8k, P=0.5 + λ=0.3)

Resume: `atom_native_step_4868000_d32_ms1M_lrd_b4828.pt`. Exact live recipe from
`scripts/train_until_coherent.sh` (stream-skip 4372000, d32, max-span 1,
field-obligatory-hard, no-merge, no-payload-copy, last-atom-readout,
efference-every 10, atom-flush 64, episode-length 8000 --episode-reset,
field-loss-weight 0.05, contrast 0, LR 1.5e-5 / SLR 3.6e-5, seed 20260913,
`--allow-parallel-train`) **plus** `--last-atom-dropout 0.5` **plus**
`--field-leak 0.3`. Output dir `checkpoints/byte_tick_fieldsat/` (live
`run_metrics.json` never touched). Baseline: ladrop 4876k (same recipe without
the lever).

### Comparison (lever is the only difference)

Val = stream ring, the same 256 transitions for both rows (`val_ablation.txt`;
the ladrop row reproduces the previous probe exactly). Field diagnostics =
`tools/diag_field_saturation.py`, 8k-tick episode from reset.

| metric | ladrop 4876k (P=0.5) | **fieldsat 4876k (P=0.5 + λ=0.3)** |
|--------|---------------------:|-----------------------------------:|
| val loss (normal) | 0.5363 | **0.4938** |
| val byte CE (normal) | 0.5309 | **0.4928** |
| **field-alone byte CE** (last_atom zeroed) | 3.399 | **3.094** |
| train: fraction of ticks at cap (`field_limited_fraction`) | 0.658 | **0.000** |
| train: max rms before clamp | 3.003 | 0.005 |
| diag: fraction of episode at cap / first cap tick | 0.633 / 2937 | **0 / never** |
| diag: temporal cos α(t, t+64) | 1.000 | 0.706 |
| inter-context α cos (6 × 64-byte forks, P=3000) | 1.000 | **0.780** |
| inter-context `a_hat` cos (what α-MLP reads) | 1.000 | **0.698** |
| linear probe next-byte CE from `a_hat` (frozen) | 3.19 | **1.99** |
| linear probe current-byte CE from `a_hat` | 3.18 | 0.42 |
| median train loss | 2.98 | 2.83 |
| seeded chat: unique replies /27 | 15 | 16 |
| seeded chat: distinct replies per prompt triple | 1.67 | 1.78 |
| seeded chat: first word | Je 27/27 | Je 25, Avec 2 |
| seeded chat: ':' fraction of chars | 0.40 | 0.23 |

References on this window: unigram ≈ 3.24, bigram ≈ 1.77, uniform 5.55.

Read it plainly:

* **The field is off the cap.** 0 of 8000 train ticks clamped (was 66%). α
  now changes with the text: fork cosine 1.000 → 0.78, and temporal cosine over 64
  ticks 1.000 → 0.71.
* **The field holds context, but the readout has not caught up.** A frozen linear
  probe on the same `a_hat` gets next-byte CE **1.99**, close to bigram 1.77. The trained
  α-MLP path alone reaches only **3.09**. That beats the unigram (3.24) and
  ladrop (3.40) by 0.3 nats, but it is nowhere near the probe. After 8k steps at SLR 3.6e-5 the readout is the bottleneck,
  not the field.
* Normal val improved (0.536 → 0.494). Train loss improved slightly.
* **Chat is still not coherent.** Replies are still attractor soup
  (`Je … ??isssistant::::…`). They are still almost independent of the prompt: for
  a given seed and T the three prompts mostly give the same reply. This is expected with
  a ~3-tick memory (λ=0.3), because the prompt is forgotten within a few bytes of the reply.
  There are fewer `:` runs, and 2/27 replies open with `Avec`.

### Chat (exact)

Coherence probe (unseeded, first line, T=0.8, top-k 12, 96 packets):

| prompt | ladrop 4876k | fieldsat 4876k |
|--------|--------------|----------------|
| Bonjour, comment ça va ? | `Bonne` | `Oui,   ??isssistant::::::::::::::::::roprrirraainuonne suis` |
| Qui es-tu ? | `Bonne` | `Avec   ??isssistant::::::::::::::::::roprrirraainuonne suis` |
| Il était une fois | `Bonne` | `Avec   ??isssistant::::::::::::::::::roprrirraainuonne suis` |

(runs of spaces shortened here; full text in `gen_coherent_4876000_fieldsat.txt`.)
Seeded examples (`chat_seeds_fieldsat_4876000.txt`): T=0.8 s=1 `Qui es-tu ?` →
`Je   D'approposes  soir ré réchir.`; T=0.8 s=3 → `Je   Barle   Bonne   Jarle`;
T=1.0 s=3 → `Je   rdirraissir,   D''onn   narle   Mirr`.

## Verdict

**Technical PASS on the saturation hypothesis. Weak on the success signal. Not fluent.**
Removing the pin (λ=0.3) makes α text-dependent and puts near-bigram
information into `a_hat` (probe 1.99). In 8k steps the α-MLP converts only a
little of it: field-alone CE 3.40 → 3.09, "below 3.40" but not "clearly toward
1.8". Chat is unchanged in quality.

## Next

1. **Continue this branch (P=0.5, λ=0.3) for ≥ 40k steps** and track field-alone
   CE against the 1.99 linear-probe ceiling. If it stalls near 3, raise **only the
   α-MLP LR** (`alpha_byte_proj`) or give it a warm-start fit. The field is now
   informative, and the readout is what lags.
2. A λ = 0.1 arm, for longer memory. The frozen probe was 2.45, but it keeps about 10 bytes, which may help
   prompt-conditioning in chat. λ = 0.3 forgets the prompt in about 3 bytes.
3. Do not merge λ into the live loop until field-alone CE is below about 2.5. The live loop is
   untouched: the default is λ = 0 (bit-identical, tested).

## Artifacts

`docs/artifacts/field_saturation/`: `diag_ckpts_fieldsat.json` (live+ladrop),
`diag_sim_leak_fieldsat.json` (frozen λ sweep), `diag_fieldsat_4876000.{json,log}`,
`injection_stats.py` + `injection_stats_live_then_ladrop.jsonl`,
`val_ablation.{json,txt}`, `chat_seeds_fieldsat_4876000.{json,txt}`,
`gen_coherent_4876000_fieldsat.txt`, `run_metrics_4876000_fieldsat.json`,
`train_fieldsat.log`, `pytest_full.txt`, `pytest_field_leak.txt`,
`measure.sh`, `run_yield.sh`, `yield_to_live.py`, `yield_log.txt`.

## CPU note

Everything ran with `OMP_NUM_THREADS=1` under `run_yield.sh`/`yield_to_live.py`.
Jobs start SIGSTOPped and receive SIGCONT only while the live chunk is in its single-threaded stream-skip
phase. The live loop was never signalled. Live chunk throughput during this work was
72.8 / 67.6 / 70.6 / 40.8 / 70.8 steps/s (chunks 4900k…4932k), against 67–76 before. The
40.8 chunk (4916k→4924k, trained ≈14:41–14:44 ET) started after the probe had already
exited (14:38 ET), while `measure.sh` was SIGSTOPped. In that window a sandbox host
process (`sand-host` node + `codebase-telemetry`, not part of this work) was measured at ≈100% CPU.

## Branch continuation (running)

`scripts/train_fieldsat_branch.sh` continues the branch from the 4876k probe checkpoint to 4916k in 8k chunks.
It uses the same recipe, and each chunk saves
`checkpoints/byte_tick_fieldsat/atom_native_step_<END>_d32_ms1M_lrd_b4828_fieldsat.pt`.
The script can be resumed: it restarts from the newest branch checkpoint, and an interrupted chunk is re-run.
It uses `THREADS` (default 1) and `YIELD` (default 1, self-SIGSTOP while the live chunk is multi-threaded).
After each chunk it appends one row to `docs/artifacts/field_saturation/branch_progress.tsv` with these metrics:
* normal val and field-alone CE on the fixed 4876k ring, cached in
  `checkpoints/byte_tick_fieldsat/val_ring_4876000.pt` via the new opt-in
  `tools/eval_last_atom_ablation.py --ring-cache`;
* the frozen linear next-byte probe on `a_hat`;
* the fraction of ticks at the cap (train + diagnostic);
* fork cosine;
* the 3-prompt coherence chat.

Per-chunk JSON goes in `docs/artifacts/field_saturation/branch/`. The master log is
`logs/fieldsat_branch_master.log`.
