# Byte-tick — one atom = one UTF-8 byte

English. Numbers. No fluency claim until a probe shows readable FR.

## Why

Spans of 5–32 bytes made train and generate different graphs:

- train: `P(whole span | α after true packet)`
- generate: independent argmax per slot, then commit

Language needs `P(byte_t | bytes_<t, field)`. ATOM is already tick-based.
The missing contract is: **one tick, one byte, one atom**.

## Single mechanism

`max_span_bytes = 1` and `max_payload_bytes = 1`.

Same `forward_packet` / `transition_loss` / `generate_packets` path.
No new module. No attention. No HF tokenizer. No MERGE retune. No new aux.

Hard mix on this tree:

- logits = α-MLP(α) + 0.3 frozen JL(α) + last-atom
- no `cap ≤ frozen RMS`
- `packet_from_payload` infers `newline|punct|whitespace|max_span` (never `"generated"`)
- payload-copy **OFF** at train and chat
- speech gate runs on the **accumulated** phrase, not on a 1-byte packet

## Banned in this cut

Attention · vocab farm · deeper MLP stack · printable-aux retune ·
linguistic span 16→32 · payload copy · L_ign · changing MERGE threshold ·
declaring fluent · overwriting the only good snapshot in-place · `.pt` in git

See `docs/MEASURE_LOG.md` for the d=16 numbers.
