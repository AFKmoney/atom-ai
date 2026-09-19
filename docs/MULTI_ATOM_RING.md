# Multi-atom payload ring (atom_flush_every=0) — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Prior (required)

Atom-payload prod @ `b53ef2c`: hard+MERGE thr=0.45 + payload prod +25k @
3.458M→**3.483M**; logits cos **0.999**; chat still noise. Falsified cause:
`atom_flush_every=256` left one shared role-prime atom — copy-bias homogenized.

Do **not** reopen `byte_decoder` / new aux losses / MERGE thr without evidence.

## Mechanism (one) — retain payload ring

| piece | detail |
|-------|--------|
| change | `--atom-flush-every` **default 256→0** (0=never clear structural atoms) |
| keep | hard ON; MERGE thr=**0.45**; payload prod; nxp=0.05; linguistic@32 |
| intent | living atoms keep MERGE-concat ring (last ~128B) so α-gated copy-bias sees prompt-distinct bytes |
| banned | byte_decoder reopen; new aux losses; MERGE retune; attention/HF |

Unit tests unchanged (payload prod still green).

## Data / train

| item | value |
|------|-------|
| resume | hard/chat ckpt @ **3,483,000** |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | **+15k** (3.483M→**3.498M**) — short measure |
| hyperparams | hard ON, MERGE ON thr=0.45, nxp=0.05, **flush=0**, d=64, `field_max_rms=3` |
| throughput | ~34 tick/s CPU |

## Trajectory (3.483M → **3.498M** = +15k)

| stage | total step | atoms (log) | merges (cum) | logits cos | α cos | chat |
|-------|------------|-------------|--------------|------------|-------|------|
| payload end | 3,483,000 | 1 | 57772 | 0.999 | 0.645 | noise |
| flush=0 +15k | **3,498,000** | **1** (all steps) | ~72k+ | **0.999** | **0.616** | still noise |

Flush disabled as intended (`atom_flush_every=0` in metrics). MERGE still
collapses to **atoms=1** every step — ring is one mega-atom, not multi-atom.

## Chat samples after +15k (honest — **not coherent**)

- **Bonjour** → `stu…sudteur:i  ngliit …` scraps — no phrase
- **Qui es-tu ?** → empty — not an answer
- **Il était une fois** → `A / asc aate: …` scraps — not a story opener

**No fluency claim.**

## Probe

| metric | value |
|--------|-------|
| logits off-diag cos | **0.9987** |
| α off-diag cos | **0.616** |
| reconstruction | True (partial) |
| verdict | field separates; surface still homogenized; flush=0 insufficient |

Artifacts: `docs/artifacts/multi_atom_ring/`.

## Verdict

**FAIL / STOP** — still noise after +15k; logits cos≪0.99 **not** achieved
(still ≈0.999). Disabling flush does not restore multi-atom diversity under
operational MERGE (atoms=1 throughout).

Do **not** auto-stack another +15k/25k. Next single hypothesis (one sentence):
**Disable MERGE on chat/probe ingest (train MERGE stays at thr=0.45) so prompt
packets remain distinct living atoms for α-gated payload copy-bias.**
