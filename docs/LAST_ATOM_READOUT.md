# Last-atom readout — one hypothesis after byte-tick

English. Numbers. No fluency claim.

Last living atom payload (last byte) + α-gate → next-byte logits.
Not payload_produce copy-bias. Not byte_decoder. Not attention.

```
emb = byte_embed(last_byte)
h = GELU(emb + mix(sigmoid(α_gate)) + phase(cos φ, sin φ))
logits += last_atom_head(h)
```

Dentate (bridge #1) expands concat(last, prev) and keeps top-25% winners
before the down-projection.

See `docs/MEASURE_LOG.md`.
