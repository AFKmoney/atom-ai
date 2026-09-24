#!/usr/bin/env bash
set -euo pipefail
cd /workspace/exports/atom-ai
PID=2509502
LOG=logs/lascale_adapt0_probe/post_train.log
exec >>"$LOG" 2>&1
echo "=== post_train start $(date) waiting for $PID ==="
while kill -0 "$PID" 2>/dev/null; do
  last=$(grep -oE 'run [0-9]+/8000' logs/lascale_adapt0_probe/train_3816000_3824000.log 2>/dev/null | tail -1 || true)
  echo "$(date +%H:%M:%S) waiting $last"
  sleep 120
done
echo "=== process exited $(date) ==="
sleep 5
CKPT=checkpoints/byte_tick/atom_native_step_3824000_d32_ms1M_lascale_adapt0.pt
if [[ ! -f "$CKPT" ]]; then
  alt=$(ls -1t checkpoints/byte_tick/atom_native_step_*_d32_ms1M_lascale_adapt0.pt 2>/dev/null | head -1 || true)
  echo "primary missing; alt=$alt"
  CKPT=${alt:-}
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  echo "NO_CHECKPOINT"
  touch logs/lascale_adapt0_probe/POST_FAIL
  exit 1
fi
echo "CKPT=$CKPT"
ls -la "$CKPT"
PYTHONPATH=. .venv/bin/python tools/probe_speech.py --checkpoint "$CKPT" \
  | tee logs/lascale_adapt0_probe/after_speech.txt
for p in "Bonjour, comment ça va ?" "Qui es-tu ?" "Il était une fois"; do
  safe=$(echo "$p" | tr ' /?' '___' | tr -cd 'A-Za-z0-9._-')
  PYTHONPATH=. .venv/bin/python tools/chat_atom_native.py --checkpoint "$CKPT" --prompt "$p" \
    | tee "logs/lascale_adapt0_probe/after_chat_${safe}.txt"
done
PYTHONPATH=. .venv/bin/python logs/frroll_probe/measure_la_alpha_rms.py --checkpoint "$CKPT" \
  | tee logs/lascale_adapt0_probe/after_la_alpha_rms.txt
tail -50 logs/lascale_adapt0_probe/train_3816000_3824000.log | tee logs/lascale_adapt0_probe/after_train_tail.txt
touch logs/lascale_adapt0_probe/POST_OK
echo "=== post_train done $(date) ==="
