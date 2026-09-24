# Clean-tip free-run-aux + adaptive S_min=0 probe — 2026-09-24

English. Numbers. **No fluency claim.**

## Why

Prior S clamps (fixed 0.15, adaptive floor 0.05, S_min=0) all ran on the
lascale/frroll soup lineage. True RMS match worked (la/α_eff=1.00) but speech
stayed letter-soup. Hypothesis: **lineage contamination** — re-test the same
lever from a clean live `*_lrd.pt` tip, not from `*_lascale_adapt0.pt`.
**Do not tweak S further.**

## Recipe

- Base tip: `atom_native_step_4260000_d32_ms1M_lrd.pt` (highest live tip at launch;
  4252000 was measured first then live advanced mid-setup)
- +8k → `atom_native_step_4268000_d32_ms1M_lrd_fradapt.pt`
- free-run-aux every=10 H=4 weight=1.0
- `--last-atom-scale-adaptive` (S_min=0 default, S_max=1.0)
- ms1M d32 hard: field-obligatory-hard, no-merge, no-payload-copy, last-atom-readout,
  efference-every 10, lr=3e-5 / surface=7.2e-5, episode 8000 reset, stream-skip=3764000
- `--allow-parallel-train` — live `train_until_coherent` / `*_lrd` **not** killed

## Table

| metric | before 4260k (fixed S=1) | preview adaptive S_min=0 @ freeze | after +8k fradapt |
|---|---|---|---|
| speech PASS | False (Peut-- attractor) | — | True (::: colon soup) |
| distinct | False | — | True |
| chats | empty | — | `i :e  s` / `i :e  t` / `i :::::` |
| la/α mean (eff) | 6.45 | **1.00** | **1.00** |
| la raw RMS mean | ~54 | ~81 | ~122 |
| mean S_eff | 1.00 (fixed) | 0.102 | 0.024 |
| attractor | `Peut--` + empty chat | — | `::::::` / colon-letter scraps |

## Chat samples (after)

1. Bonjour, comment ça va ? → `i :e  s`
2. Qui es-tu ? → `i :e  t`
3. Il était une fois → `i :::::`

## Honest read

- **Math succeeded again** on a clean tip: S_eff tracked below 0.05; effective
  la/α stayed 1.00 after +8k.
- Clean tip started healthier than soup lineage (la_raw ~54–81 vs ~230–458) but
  free-run-aux still inflated raw (~81→~122) and speech moved from empty/`Peut--`
  to colon soup — different attractor, still **not French sentences**.
- **Lineage-contamination hypothesis falsified for fluency on this +8k probe:**
  free-run-aux + adaptive S_min=0 from clean live `*_lrd` does not buy French.
  Next lever is elsewhere (not more S clamp tweaks).

## Live train

Untouched. After probe: tip advanced 4260k→4268k→4276k; `train_until_coherent.sh`
still cycling toward 4284k with `--allow-parallel-train`.
