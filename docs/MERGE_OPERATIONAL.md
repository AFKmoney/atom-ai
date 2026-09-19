# MERGE operational under hard-v2+ — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Falsified prior (do not reopen)

Linguistic spans 16→32 @ `02891ff` (3.325M→3.425M): mean pkt ~19 B, chat still
noise. Do **not** burn another span/printable chunk alone.

## Root cause (logged)

Train traj under linguistic +100k: `merges=0` always, `mph≈0.500` always,
`n_pairs_above_energy_floor` thousands. So **energy floor was not the block**.

Checkpoint atoms: `E≡0.5`, `φ≈0` → `coherence = cos(Δφ)·mean(√(Ei·Ej)) = 0.5`.
Default `phase_coherence_threshold=0.55` used `score < thr` → every pair rejected.

## Mechanism (one) — lower MERGE coherence gate

| knob | old | new default |
|------|-----|-------------|
| `phase_coherence_threshold` | 0.55 | **0.45** |
| `merge_energy_floor` | 0.08 | 0.08 (unchanged; pef already high) |

CLI: `--merge-coherence-threshold` / `--merge-energy-floor` (CLI wins on resume).

Keep instrumentation: `max_phase_coherence` (mph), `n_pairs_above_energy_floor`.

Also required for live MERGE (latent until merges fired):

1. Restore MERGE outputs to AtomCompiler shapes `(1, d)` / rho `(1, 1)` so the
   atom collection can stack with live atoms.
2. Consolidation: expand singleton atom-mean `E`/`stability` to `n_modes` before
   boolean mask on α (mask shape bug tripped once densified atoms boosted E).

Unit tests: synthetic coherent atoms → `merge_count > 0` under defaults; mph=0.5
regime merges at 0.45 and **not** at 0.55; `set_merge_thresholds` propagates.

## Data / train (smoke)

| item | value |
|------|-------|
| resume | `checkpoints/atom_native_obligatory_hard/` @ **3,425,000** |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | **+8k** smoke (protocol ~5k–10k) |
| hyperparams | hard ON, `L_ign=0`, printable_aux=**0**, `--enable-merge`, thr=**0.45**, floor=0.08, linguistic@32, `--no-episode-reset`, d=64, `field_max_rms=3` |
| throughput | ~39 tick/s CPU |

## Trajectory (3.425M → **3.433M** = +8k)

| stage | total step | merges (cum) | mph | atoms (log) | logits cos | α cos | chat |
|-------|------------|--------------|-----|-------------|------------|-------|------|
| linguistic end | 3,425,000 | **0** | ~0.500 | grow/flush | 0.866 | 0.713 | noise |
| MERGE smoke +8k | **3,433,000** | **7968** | ~0.69 | **1** (collapse) | **0.860** | **0.705** | still noise |

Log: `merges=` column is **cumulative** `merge_count_total`. 160/161 traj rows
had `merge_count>0` this run (~1 merge/tick after first). After each inject,
pair merges immediately → list collapses to **1** heavier atom (`ρ` bumped,
`E` boosted → mph ~0.693).

## Chat samples after +8k (honest — **not coherent**)

- **Bonjour** → binary / Latin scraps / loops — no phrase
- **Qui es-tu ?** → empty / near-empty — not an answer
- **Il était une fois** → short scraps — not a story opener
- **Bonjour (deterministic)** → `AAa.Mtl…` loop — not FR

**No fluency claim.** MERGE is operational (count > 0); speech still locked.

## Probe (after smoke)

| metric | value |
|--------|-------|
| logits off-diag cos | **0.860** |
| α off-diag cos | **0.705** |
| reconstruction | True (partial/diagnostic) |
| verdict | field carrying structure; gen still noise |

Artifacts: `docs/artifacts/merge_operational/`.

## Verdict / next

**PASS** for Step 1 mechanism: MERGE fires under hard-v2+ with thr=0.45.
Honest side effect: structural list collapses to ~1 atom/tick — densify works
as count, not as a rich multi-atom hierarchy yet.

Next (only after this commit): ONE field→next-packet production loss (small
weight), separate train — **not** stacked in this blob.
