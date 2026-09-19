# ATOM status snapshot — 2026-09-19

Repo: https://github.com/AFKmoney/atom-ai
English docs only. **No fluency claim.**

## main (this push) — speech contract, not 3.5M resume

The 3.4M–3.5M payload-copy line **stays archived below**. This push is a
**separate d=16 byte-tick line** that repaired the generate contract.

What landed in code:

- Atomizer `max_span_bytes=1` (byte-tick)
- Last-atom readout + last-atom **φ** (oscillatory binding, corpus bridge #20)
- Dentate 2-gram sparse expand (bridge #1)
- Efference copy every 10 train ticks (bridge #9), replay **gated off** by default
- `packet_from_payload` infers train-like boundaries (never `"generated"`)
- N-gram anti-repeat in `generate_packets`
- Hard mix: `α-MLP + 0.3 frozen JL + last-atom` — **no RMS cap**

Measured log: `docs/MEASURE_LOG.md`.
Bridges map: `docs/NEURO_BRIDGES_ATOM.md`.
Byte-tick protocol: `docs/BYTE_TICK.md`.

Honest chat @ 32k + anti-repeat: `Je Jestiste` / `Je pe te pe ?` — French
debris, not sentences. Best CE @ 36k-big: **1.61**.

`.pt` not in git.

## Next (one at a time)

1. Stream the 2.2 MB CATIE FR file on a real box from `36k-big` / `39.5k-hf`.
2. Do **not** reopen payload-copy, MERGE retune, or 3.5M resume for speech.
3. Do not declare fluent.

## Banned

HF tokenizer on live path · attention/DDP/vocab farm · field wipe for CE ·
stacking multiple new mechanisms in one train · declaring fluent ·
force-push · `.pt` in git.

---

# Archive — 2026-09-18 3.5M line (do not mix with byte-tick)

The previous STATUS body (payload-copy / MERGE / 3.508M / logits 0.999)
is kept in git history and in the speech-era docs that falsified those
levers (`docs/ATOM_PAYLOAD_PROD.md`, `docs/ALPHA_LOCAL_COPY.md`,
`docs/LINGUISTIC_SPANS.md`). Do not resume that line for speech.
