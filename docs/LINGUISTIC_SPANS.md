# Linguistic spans (Atomizer pack) under hard-v2+ — 2026-09-18

English. Numbers. **No fluency / GPT claim.**

North star: readable French phrases via persistent toroidal field + Atomizer
(CPU continuum only) — **no** attention / HF / DDP / clusters / Transformers.

## Falsified prior (do not reopen)

Printable aux @ `be25374` (3.275M→3.325M): prn 0.87→0.89, chat still scraps.
Do **not** burn more CE on printable aux alone.

## Mechanism (one) — longer dialogue-aligned linguistic spans

Hypothesis: eager WS flush (mean ~5 B/packet @ `max_span=16`) locked hard α-MLP
on 1–3 byte scraps. Raise `max_span_bytes` **16→32** and pack with
`pack_mode=linguistic`: accumulate toward max, flush on newlines
(`Utilisateur:` / `Assistant:` lines), prefer WS/punct cuts — still byte-span
Atomizer (**no** BPE/HF). Keep `--field-obligatory-hard` + α-MLP; no
`byte_decoder` bypass; **no MERGE threshold change**; `printable_aux_weight=0`.

### Atomizer `pack_mode=linguistic`

| rule | behavior |
|------|----------|
| newline | flush immediately (dialogue line bound) |
| WS / punct | soft cut; flush when buffer ≥ ~¾ `max_span_bytes` |
| at `max_span_bytes` | cut at last WS/punct if possible; else UTF-8 max cut |
| eager (legacy) | flush every boundary (tests / ablations) |

CLI: `--max-span-bytes 32` + `--atomizer-pack linguistic` (defaults).

### Resume / span-head pad-migrate

Hard ckpt @ **3,325,000** had `max_payload_bytes=16`. Growing to 32 pad-copies
prefix rows of surface span heads (α frozen maps, α-MLP `fc2`, decoders, field
skips); **new rows only** at init; field/core kept. Flag:
`training["span_head_migrated"]`.

## Data / train

| item | value |
|------|-------|
| resume | `checkpoints/atom_native_obligatory_hard/` @ **3,325,000** |
| data | `--stream --data-glob data/dialogue_shards/part_*` |
| budget | +50k + probe/chat; second +50k (protocol max); then **STOP** |
| hyperparams | hard ON, `L_ign=0`, printable_aux=**0**, `--no-episode-reset`, d=64, `field_max_rms=3`, MERGE defaults unchanged |
| throughput | ~35 tick/s CPU |

## Trajectory (3.325M → **3.425M** = +100k)

| stage | total step | logits cos | α cos | mean pkt B | chat (honest) |
|-------|------------|------------|-------|------------|---------------|
| printable end | 3,325,000 | **0.912** | **0.837** | ~5 (eager@16) | scraps |
| linguistic +50k | **3,375,000** | **0.882** | **0.737** | ~19 (ling@32) | still noise / loops |
| linguistic +100k | **3,425,000** | **0.866** | **0.713** | ~19 (ling@32) | still noise / empty |

Probe after +100k: logits **≪0.99** (0.866), α alive (0.713). No soft-collapse.
Packet count per MB dropped (~97k→~27k) — spans really lengthened. Speech still
locked.

## Chat samples after +100k (honest — **not coherent**)

- **Bonjour** → binary/Latin scraps / loops — no phrase
- **Qui es-tu ?** → empty / near-empty — not an answer
- **Il était une fois** → short scraps — not a story opener
- **Bonjour (deterministic)** → `AAa.M` / `l,,…` loop — not FR

**STOP verdict:** after **+100k** linguistic-span steps (protocol max),
generations do **not** show multi-word readable French. Longer CE targets alone
did not unlock phrase-level FR under hard-v2+ α-MLP. Do **not** stack another
span/printable chunk. No fluency claim.

## Next hypothesis (single) — MERGE densify

Train logs: `merges=0`, `mph≈0.5` throughout linguistic runs. Next lever:
**MERGE densification** (threshold / schedule only after mph evidence) so the
field builds heavier dialogue structure — **not** another span or printable
aux stack. Do not retune MERGE in the same change as a new readout lever.

## Artifacts

`docs/artifacts/linguistic_spans/{pytest.txt,chunk_50k/,chunk_100k}/`
(train excerpts, probes, chat, run_metrics). Train logs:
`logs/linguistic_spans_chunk_{50k,100k}.txt`.

## Banned

Attention · HF tokenizer on live path · CE bypass via byte_decoder · stacking
printable+MERGE+spans · declaring fluent · force-push · `.pt` in git · more
printable-aux CE · more blind span CE past this STOP.
