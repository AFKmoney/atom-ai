#!/usr/bin/env bash
# Pure scaling ms1M chunks: START→1500k, 8k steps each. Exact recipe, 0 patches.
set -euo pipefail
cd /workspace/exports/atom-ai
export PYTHONPATH=.
PY=.venv/bin/python
OUT=checkpoints/byte_tick
BASE=496000   # ms1M line continued from s21 tip; skip = start - BASE (matches prior logs)
END_TARGET=1500000
START=${1:-736000}

while [ "$START" -lt "$END_TARGET" ]; do
  END=$((START + 8000))
  if [ "$END" -gt "$END_TARGET" ]; then END=$END_TARGET; fi
  STEPS=$((END - START))
  SKIP=$((START - BASE))
  if [ "$SKIP" -lt 0 ]; then SKIP=0; fi
  # after full 500k packets into stream, wrap like 2e epoch guidance
  # keep historical skip=start-496k which grows; stream loop-shards handles wrap
  RESUME="$OUT/atom_native_step_${START}_d32_ms1M.pt"
  CKPT_NAME="atom_native_step_${END}_d32_ms1M.pt"
  LOG="logs/train_d32_${START}_${END}_ms1M.log"
  if [ ! -f "$RESUME" ]; then
    echo "MISSING resume $RESUME" >&2
    exit 1
  fi
  if [ -f "$OUT/$CKPT_NAME" ]; then
    echo "skip exists $CKPT_NAME"
    START=$END
    continue
  fi
  echo "=== $(date -Is) train $START->$END skip=$SKIP steps=$STEPS ==="
  $PY -u tools/run_atom_native.py \
    --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 \
    --stream-skip-packets "$SKIP" \
    --resume "$RESUME" \
    --output-dir "$OUT" \
    --checkpoint-name "$CKPT_NAME" \
    --steps "$STEPS" \
    --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 \
    --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout \
    --efference-every 10 --learning-rate 5e-5 --surface-learning-rate 1.2e-4 --seed 20260913 \
    --atom-flush-every 64 --episode-length 8000 --episode-reset --slow-every 1 \
    --field-loss-weight 0.05 --field-contrast-weight 0 \
    --allow-parallel-train \
    > "$LOG" 2>&1
  echo "=== done $END ==="
  # light measure
  if [ -f "$OUT/run_metrics.json" ]; then
    cp "$OUT/run_metrics.json" "$OUT/run_metrics_${END}_d32_ms1M.json" || true
  fi
  START=$END
done
echo "REACHED $END_TARGET"
