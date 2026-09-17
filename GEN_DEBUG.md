# Generation debug — 2026-09-16/17 (PT)

## Root causes

1. **Surface bias drowned the field (primary)**  
   After priming, toroidal `_output_state` had tiny amplitude (`std ~ 0.003`).  
   Measured `|W x| / |b| ≈ 0.005` on `AtomSurfaceHead.byte_decoder`.  
   Byte-logit cosine between unrelated prompts was **1.000** — every prompt
   greedily decoded to the same first packet (`\ne  `). Chat looked like
   embryonic `stant:` / `ant:` fragments only later as the field grew, not
   because the prompt was read.

2. **Episode reset every 64 packets**  
   `tools/run_atom_native.py` wiped field / consolidation / atoms each episode.
   Dialogue spans are much longer; persistence never survived a turn.

3. **Generated packet boundary tag**  
   `packet_from_payload(..., boundary="generated")` used a feature code unseen
   in training (`encode` emits whitespace/punct/newline/max_span). Mild
   distribution shift on the feedback path.

4. **Sampling noise**  
   Chat defaults (`temperature=0.9`, `top_k=8`) amplified an already collapsed
   head. Not the root cause, but made UTF-8 look worse.

## Fixes

| Change | Where |
|--------|--------|
| `LayerNorm` before surface linears | `AtomSurfaceHead` |
| Legacy checkpoint: load without `state_norm`, damp biases `×0.05` | `AtomNativeModel.load` |
| Infer structural boundary on generated payloads | `Atomizer.packet_from_payload` |
| Printable soft bias + safer sampling | `AtomSurfaceHead.decode` |
| Full-prompt priming docs; optional `reset=` | `generate_packets` |
| Default `episode_length=512`, `--episode-reset` off (BooleanOptional) | `run_atom_native.py` |
| Soft atom-list clear near `n_atoms_max` without wiping field | training loop |
| Cumulative resume step | checkpoint `training.step` |
| Role priming `Utilisateur:` / `Assistant:` | `chat_atom_native.py` |
| Regression test + smoke script | `test/test_generation_smoke.py`, `tools/smoke_gen_check.py` |

## Training continuity

```bash
PYTHONPATH=. .venv/bin/python tools/run_atom_native.py \
  --data data/corpus_train_dialogue.txt \
  --output-dir checkpoints/atom_native_chat \
  --resume checkpoints/atom_native_chat/atom_native.pt \
  --steps 300000 --d-model 64 --n-modes 64 --n-atoms-max 512 \
  --max-span-bytes 16 --episode-length 512 --no-episode-reset \
  --field-max-rms 4.0 --learning-rate 3e-4 --log-every 100
```

`--no-episode-reset` is the default; pass `--episode-reset` for the old wipe.

## Honesty

LayerNorm + bias damp **changes** the surface operating point. Expect a short
loss bump then recovery. Chat will not become GPT-quality; success means
prompt-sensitive, mostly printable UTF-8 with dialogue-ish fragments.

### Speed note
With `--no-episode-reset`, the **atom list** is still cleared each episode window so aggregation stays cheap; **field alpha + consolidation** persist.

## Chosen continued-train settings (this session)

Full `--no-episode-reset` saturated `field_rms` at the cap within ~1k steps and
kept hundreds of atoms → ~20 transitions/s. Practical compromise that still
fixes the “thrash every 64” problem:

- `--episode-length 512` (8× old default)
- `--episode-reset` **on** (clean field each window; atoms stay cheap)
- LayerNorm surface migration + bias damp still applied on load
- `lr=3e-4`, `steps=250000`, dialogue corpus

Atom-list-only flush without field wipe remains available via `--no-episode-reset`
for experiments that can afford the slower field-saturated regime.

## Session results (2026-09-17 ~02:00 PT)

### Before fix
- All prompts greedily decoded to the same first packet (`\ne  `).
- `|Wx|/|b| ≈ 0.005`; byte-logit cosine across prompts = 1.0.
- Chat log showed embryonic `stant:` fragments only.

### After code fix (pre-retrain smoke)
- Prompts diversified immediately after LayerNorm + bias×0.05 migration.
- Still digit/symbol garbage (operating point shifted; needs retrain).

### After +250k retrain (total steps 650k)
- Log: `logs/train_after_genfix_250k.log`
- Train: first loss ≈13.8 (migration bump) → final step loss low; val byte_ppl ≈31.3
- Throughput ≈124 transitions/s with `episode_length=512`, `--episode-reset`, `--atom-flush-every 64`
- Checkpoint `energy_decay` had drifted to **-21**; repaired to 0.5 and default clamp added in code.
- Smoke: mostly printable UTF-8, prompt-diverse, longer than 16 chars.
- FR chat samples still **not coherent**, but start with dialogue-ish tokens (`Je`, spaces, `?`) instead of identical collapse.

### Honest verdict
Generation path + episode length fixes address the concrete bugs. Conversational French is still embryonic byte-LM noise — not assistant-quality.


## Decay clamp follow-up (2026-09-17 ~02:20 PT)

1.05M persist left `energy_decay≈0.001` (weak default floor `1e-3`).  Code now
always clamps to **`[0.3, 0.95]`** every step + on load; see **DECAY_FIX.md**.
Retrain → `checkpoints/atom_native_chat_persist_v2/` with `field_max_rms=3.0`
and optional surface LR group.
