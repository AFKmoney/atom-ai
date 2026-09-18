# Hard-v2+ deeper α-only MLP decode — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Mechanism (one) — deeper α-only under hard

Hypothesis after hard-v2 +200k (logits **0.916**, chat short-token noise):
bottleneck = under-capacity of Linear `α→byte` map under hard — **not** a CE
bypass. Deepen the α-only trainable adapter; keep frozen JL residual + RMS
cap; still **no** `byte_decoder` / skip bypass.

`--field-obligatory-hard` contract unchanged; trainable adapter upgraded:

1. `AlphaOnlyMLP`: `α_feat (JL fingerprint) → Linear(4·d) → GELU → Linear(out)`
   for bytes and length (replaces single Linear `alpha_byte_proj` /
   `alpha_length_proj`)
2. Forward still:
   `logits = frozen_α_map(α) + cap * obligatory_scale() * α_MLP(α)`
   with `cap = min(1, frozen_rms / trainable_rms).detach()`
3. Non-α decoder / skip / gates stay **zeroed + frozen**
4. Load: Linear→MLP migrate reinits **only** the MLP; frozen JL + maps kept
   from hard-v2 ckpt (`alpha_mlp_migrated`)

Unit tests (`test/test_field_obligatory_hard.py`): decoder frozen; α vs zeros
distinct; trainable α MLP has grads; RMS cap; hidden=`4*d`.

## Data / train

| item | value |
|------|-------|
| resume | `checkpoints/atom_native_obligatory_hard/` @ **3,125,000** (hard-v2) |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| chunks | +50k × 3 = **+150k** (no early stop — no multi-word FR) |
| hyperparams | hard ON, `L_ign=0`, `--no-episode-reset`, d=64, `field_max_rms=3` |
| printable | chat/probe `prefer_printable=True` (existing) |
| throughput | ~34–36 tick/s CPU |

## Trajectory (3.125M → **3.275M** = +150k)

| stage | total step | logits cos | α cos | chat (honest) |
|-------|------------|------------|-------|---------------|
| hard-v2 end | 3,125,000 | **0.916** | **0.842** | short-token loops |
| deeper +50k | 3,175,000 | **0.915** | **0.845** | Latin scraps / loops |
| deeper +100k | 3,225,000 | **0.916** | **0.845** | short-token noise |
| deeper +150k | **3,275,000** | **0.914** | **0.840** | short-token noise |

Probe after +150k: logits **≪0.99** (0.914), α alive (0.840). No soft-collapse.
Mild logits drift within noise of hard-v2 baseline.

## Chat samples after +150k (honest — **not coherent**)

- **Bonjour** → `u…Qsuu…suiedus a…` — letter scraps, no phrase
- **Qui es-tu ?** → multiline short tokens (`eiaq…` / `es-…`) — not an answer
- **Il était une fois** → `!ali!flm!pt-!8Qq!…` — not a story opener
- **Bonjour (deterministic)** → `u…Qsu'Qsu'…su'esu'es…` loop

**STOP verdict:** after **+150k** deeper-MLP steps, generations do **not** show
multi-word readable French. Field probe OK; surface still emits short-token
noise. Deeper α MLP alone did **not** unlock phrase-level FR in this budget.
No fluency claim.

## Artifacts

`docs/artifacts/hard_v2_deeper/{chunk_50k,chunk_100k,chunk_150k}/`
(train excerpts, probes, chat, run_metrics). Train logs:
`logs/hard_v2_deeper_chunk_*.txt`. Pytest: `docs/artifacts/hard_v2_deeper/pytest.txt`
(44 passed, 1 skipped).

## Banned

Attention · HF tokenizer on live path · CE bypass via byte_decoder · stacking
MERGE retune · declaring fluent · force-push · `.pt` in git.
