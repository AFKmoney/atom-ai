# PERSISTENCE_TEST — field farm vs Transformer regime

## One-liner

**CE down + gen still noise without field reconstruction = still Transformer regime without the farm.**

## Why this test exists

Packet CE can fall while the model behaves like a small byte-LM: the surface
head predicts local bigrams and the toroidal field is unused or collapsed.
Capability claims that rely on “persistent structured matter” require a probe
that asks whether the **field** still differs by prompt and whether past
packet information can be recovered from field/consolidation state — not only
whether CE improved.

## What we measured (this run)

| Metric | Value |
|--------|-------|
| Checkpoint | `/workspace/exports/atom-ai/checkpoints/atom_native_lign_smoke/atom_native.pt` |
| Mean field RMS after prompt | 0.005141 |
| Mean field RMS after 20 gen | 0.033300 |
| Mean off-diag cosine (surface logits) | 0.991285 |
| Mean off-diag cosine (alpha) | 0.698694 |
| Reconstruction succeeds (diagnostic) | True |
| Verdict | field is carrying structure (prompt-sensitive; diagnostic reconstruction partial) |

Full detail: `docs/FIELD_PROBE.md`, `logs/field_probe_report.json`.

## Pass / fail reading

- **Surface-only byte-LM:** inter-prompt surface cosine ≈ 1, field separation absent or unread.
- **Field carrying structure:** alpha/persistence differ across prompts; diagnostic recovery from field features beats a mean baseline; surface logits also move.
- **Transformer regime without the farm:** CE/metrics look healthy but this probe fails — scale Θ or more CE steps will not buy persistence.

## Next levers (named, not built here)

1. Real **MERGE** into durable field structures
2. **Dual clocks** (fast surface tick / slow field tick)
3. **Adaptive spans** (content-aware packet duration)

Run: `PYTHONPATH=. python tools/probe_field_persistence.py`
