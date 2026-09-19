# Speech lock repair — 2026-09-18

English. Numbers. **No fluency / GPT claim until a prompt emits a distinct phrase.**

Grok Bot cascade STOP at `48cdb19`. Do not stack another α-map / payload / +10k.

## What was broken

| fact | number |
|------|--------|
| payload + MERGE-train | logits **0.838 → 0.999** |
| α-local copy +10k | logits **0.9987**, chat `auiseateu : aernun e` |
| MERGE-off ingest | atoms 4–6, surface still identical |
| last healthy surface | field_next @ **3,458,000** logits **0.838** / spans @ **3,425,000** logits **0.866** |

Two mechanical locks, not "needs more CE":

1. `payload_produce` broadcast the winner unigram hist onto **every** span slot → letter soup.
2. `generate_packets` committed that soup back into the atomizer → the next tick copies soup.

Turning the frozen α map down on the 3.508M ckpt would have fed the collapsed payload path. Falsified.

## One repair (this commit)

- Position-aligned copy only (no broadcast hist).
- Chat `payload_copy=False` (hard path = frozen α + capped α-MLP only).
- Dialogue wrap `Utilisateur: …\nAssistant:` (train distribution).
- `speech_ok` gate: do not re-ingest soup. Hint lexicon, not a Transformer.

Banned: attention, HF tokenizer, byte_decoder reopen, +10k on alpha_local, MERGE retune in the same change.

## Run on the box (no Bot)

```bash
# STOP the Bot first.
git pull

# Prefer 3.458M field_next, then 3.425M spans. NEVER alpha_local 3.508M.
python3 scripts/speech_chat.py --prompt "Bonjour"
python3 scripts/speech_chat.py --prompt "Qui es-tu ?"
python3 scripts/speech_chat.py --prompt "Il était une fois"
```

If the response is empty, `speech_ok` refused soup. That is correct. Load the 3.458M file, not 3.508M.

## Next train (only if chat on 3.458M is still scraps)

Single lever: resume **3,458,000**, `payload_enabled=False`, chat merge-off already default, **do not** lower MERGE thr. Budget ≤25k. Stop if logits > 0.95. Gate = distinct phrases across the three prompts, not logits cos.
