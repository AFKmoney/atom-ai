#!/usr/bin/env bash
# Post-probe measurements (run under run_yield.sh; single-threaded).
set -u
cd /workspace/exports/atom-ai
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
A=docs/artifacts/field_saturation
LA=checkpoints/byte_tick_ladrop/atom_native_step_4876000_d32_ms1M_lrd_b4828_ladrop.pt
FS=checkpoints/byte_tick_fieldsat/atom_native_step_4876000_d32_ms1M_lrd_b4828_fieldsat.pt
# 1) exact coherence-probe CLI (unseeded, first line, as in train_until_coherent.sh)
G=$A/gen_coherent_4876000_fieldsat.txt; : > $G
for p in "Bonjour, comment ça va ?" "Qui es-tu ?" "Il était une fois"; do
  RESP=$($PY tools/chat_atom_native.py --checkpoint "$FS" --prompt "$p" --no-role-prime \
    --max-packets 96 --max-length 100 --temperature 0.8 --top-k 12 2>/dev/null | sed -n 's/^Response: //p' | head -1)
  echo "PROMPT: $p" >> $G; echo "RESP: $RESP" >> $G
done
# 2) seeded chats (seeds 1-3 x T 0.8/0.6/1.0, same order as the ladrop baseline file)
$PY tools/chat_seeds_probe.py --checkpoint "$FS" --seeds 1 2 3 --temps 0.8 0.6 1.0 \
  --out $A/chat_seeds_fieldsat_4876000.json > $A/chat_seeds_fieldsat_4876000.txt 2>&1
# 3) field diagnostic on the probe ckpt (its own λ) + frozen linear probes
$PY -u tools/diag_field_saturation.py --checkpoints fieldsat4876k="$FS" --episode-ticks 8000 \
  --prefixes 256 3000 --probe --out $A/diag_fieldsat_4876000.json > $A/diag_fieldsat_4876000.log 2>&1
# 4) val normal vs last_atom-zeroed, same 256-transition ring for baseline and lever
$PY -u tools/eval_last_atom_ablation.py --end-steps 4876000 \
  --checkpoints "$LA" "$FS" --out $A/val_ablation.json > $A/val_ablation.txt 2>&1
echo DONE > $A/.measure_done
