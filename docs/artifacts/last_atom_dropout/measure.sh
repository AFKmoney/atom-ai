#!/usr/bin/env bash
set -u
cd /workspace/exports/atom-ai
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
A=docs/artifacts/last_atom_dropout
LA=checkpoints/byte_tick_ladrop/atom_native_step_4876000_d32_ms1M_lrd_b4828_ladrop.pt
LIVE=checkpoints/byte_tick/atom_native_step_4876000_d32_ms1M_lrd_b4828.pt
START=checkpoints/byte_tick/atom_native_step_4868000_d32_ms1M_lrd_b4828.pt
# 1) exact coherence-probe CLI (unseeded, as in train_until_coherent.sh)
G=$A/gen_coherent_4876000_ladrop.txt; : > $G
for p in "Bonjour, comment ça va ?" "Qui es-tu ?" "Il était une fois"; do
  RESP=$($PY tools/chat_atom_native.py --checkpoint "$LA" --prompt "$p" --no-role-prime \
    --max-packets 96 --max-length 100 --temperature 0.8 --top-k 12 2>/dev/null | sed -n 's/^Response: //p' | head -1)
  echo "PROMPT: $p" >> $G; echo "RESP: $RESP" >> $G
done
# 2) seeded chats, 3 ckpts x 3 seeds x 3 temps
for tag in ladrop_4876000:$LA live_4876000:$LIVE start_4868000:$START; do
  name=${tag%%:*}; ck=${tag#*:}
  $PY tools/chat_seeds_probe.py --checkpoint "$ck" --seeds 1 2 3 --temps 0.8 0.6 1.0 \
    --out $A/chat_seeds_${name}.json > $A/chat_seeds_${name}.txt 2>&1
done
# 3) val normal vs last_atom-zeroed on both rings
$PY -u tools/eval_last_atom_ablation.py --end-steps 4868000 4876000 \
  --checkpoints "$START" "$LIVE" "$LA" --out $A/val_ablation.json > $A/val_ablation.txt 2>&1
echo DONE > $A/.measure_done
