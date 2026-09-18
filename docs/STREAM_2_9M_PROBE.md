# STREAM_2_9M_PROBE — field persistence at stream step 2.9M

_Generated: 2026-09-17 19:41:46 PDT_

English only. No fluency claim.

## Checkpoint

| Field | Value |
|-------|-------|
| Path | `/workspace/repos/toroidal-fractal-intelligence/checkpoints/atom_native_stream_dialogue/atom_native.pt` |
| Training step | **2 900 000** (stream finished) |
| Config | d_model=64, n_modes=64, n_atoms_max=512, max_payload_bytes=16, field_max_rms=3.0 |
| Bytes | 7 967 626 |

Probe: `PYTHONPATH=. .venv/bin/python tools/probe_field_persistence.py --checkpoint … --skip-docs`

Artifacts: `docs/artifacts/stream_2_9M_probe/field_probe_report.json`

## Hard numbers

| Metric | Value |
|--------|-------|
| Mean off-diag cosine **surface logits** | **0.979936** |
| Mean off-diag cosine **α** (field) | **0.706717** |
| Mean field RMS after prompt | **0.005340** |
| Mean field RMS after 20 gen | **0.026840** |
| Reconstruction succeeds (diagnostic) | True |
| Field differs across prompts | True |

### Per-prompt RMS

| Prompt | RMS after prompt | RMS after 20 gen | n_atoms (prompt) |
|--------|------------------|------------------|------------------|
| Bonjour | 0.003751 | 0.026029 | 3 |
| Qui es-tu ? | 0.008058 | 0.024798 | 7 |
| Il était une fois | 0.005801 | 0.030503 | 6 |
| Utilisateur: Bonjour\\nAssistant: | 0.003751 | 0.026029 | 3 |

## Verdict

**field is carrying structure (prompt-sensitive; diagnostic reconstruction partial)**

α separates across prompts (off-diag cosine ~0.71). Surface logits are somewhat prompt-sensitive (~0.980) but still high — decode does not cleanly express field differences. Generation remains embryonic/noisy (see chat samples). This is **not** a fluency claim.

Compare earlier L_ign series (logits stuck ~0.991–0.992): the finished stream is slightly better on logits cosine, but the surface still largely under-reads α for CE-path decoding.

## Optional chat samples (deterministic, role-primed)

Noise only — no fluency judgment:

| Prompt | Preview |
|--------|---------|
| Bonjour | `!B…j…y…?…I…y…?h…e?h…k?…` (binary/control noise) |
| Qui es-tu ? | `!B♀…y…?…♾…y…?……e?……e?…` (noise; prompt-distinct from Bonjour) |
| Il était une fois | `!B♀…y…?…♀…y…?……e?……e?…` (noise; collapses toward shared pattern) |

## Method notes

- Read-only: `model.eval()`, `requires_grad=False`, no optimizer.
- Priming matches chat: Atomizer → `forward_packet` per packet.
- Did **not** restart the finished stream marathon.
