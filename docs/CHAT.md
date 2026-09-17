# Chat — ATOM / atom-ai

### One-shot

```bash
PYTHONPATH=. python tools/chat_atom_native.py \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt \
  --prompt "Bonjour, comment ça va"
```

### Interactive

```bash
PYTHONPATH=. python tools/chat_atom_native.py \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt \
  --interactive

# or
PYTHONPATH=. python -m src.main --mode interactive \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt
```

Role priming uses `Utilisateur:` / `Assistant:` style prefixes when applicable.

### Smoke generation check

```bash
PYTHONPATH=. python tools/smoke_gen_check.py \
  --checkpoint checkpoints/atom_native_chat_persist/atom_native.pt
```

### Known limitations (read this)

1. **Not fluent.** Outputs are next-**byte-packet** predictions from a small
   toroidal field model. Expect fragments, repetition, and noisy UTF-8.
2. **Not an instruction-tuned LLM.** No RLHF, no chat template stack, no HF hub.
3. Shipped ckpt ≈ **1.05M CPU steps**, `d_model=64` — research demo scale.
4. After gen-fix, prompts should be **diverse** (not collapsed to one packet);
   diversity ≠ coherence.
5. Sampling (`temperature`, `top_k`) can make UTF-8 look worse if the surface is
   still weak — prefer greedy/smoke defaults when diagnosing.

Success today: atom-native path runs end-to-end without GPT-2. Fluency is a
**training + data** problem, not a reason to reintroduce Transformers.
