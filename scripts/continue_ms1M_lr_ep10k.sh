#!/usr/bin/env bash
# Continue ms1M with BOTH levers: LR 3e-5/7.2e-5 + episode-length 10000.
# Resume chain from lrdecay tip; after first chunk names drop _lrdecay suffix but keep recipe.
set -euo pipefail
cd /workspace/exports/atom-ai
export PYTHONPATH=.
PY=.venv/bin/python
OUT=checkpoints/byte_tick
BASE=496000
END_TARGET=1500000
# start from lrdecay 880k file once, then normal tip names
RESUME0="$OUT/atom_native_step_880000_d32_ms1M_lrdecay.pt"
START=880000
FIRST=1

while [ "$START" -lt "$END_TARGET" ]; do
  END=$((START + 10000))
  if [ "$END" -gt "$END_TARGET" ]; then END=$END_TARGET; fi
  STEPS=$((END - START))
  SKIP=$((START - BASE))
  if [ "$SKIP" -lt 0 ]; then SKIP=0; fi
  if [ "$FIRST" = 1 ]; then
    RESUME="$RESUME0"
    FIRST=0
  else
    RESUME="$OUT/atom_native_step_${START}_d32_ms1M_lrep10.pt"
  fi
  CKPT_NAME="atom_native_step_${END}_d32_ms1M_lrep10.pt"
  LOG="logs/train_d32_${START}_${END}_ms1M_lrep10.log"
  if [ ! -f "$RESUME" ]; then
    echo "MISSING $RESUME" >&2
    exit 1
  fi
  if [ -f "$OUT/$CKPT_NAME" ]; then
    echo "skip exists $CKPT_NAME"
    START=$END
    continue
  fi
  echo "=== $(date -Is) $START->$END skip=$SKIP ep=10000 lr=3e-5/7.2e-5 ==="
  $PY -u tools/run_atom_native.py \
    --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 \
    --stream-skip-packets "$SKIP" \
    --resume "$RESUME" \
    --output-dir "$OUT" \
    --checkpoint-name "$CKPT_NAME" \
    --steps "$STEPS" \
    --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 \
    --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout \
    --efference-every 10 --learning-rate 3e-5 --surface-learning-rate 7.2e-5 --seed 20260913 \
    --atom-flush-every 64 --episode-length 10000 --episode-reset --slow-every 1 \
    --field-loss-weight 0.05 --field-contrast-weight 0 \
    --allow-parallel-train \
    > "$LOG" 2>&1
  # snapshot metrics
  if [ -f "$OUT/run_metrics.json" ]; then
    cp "$OUT/run_metrics.json" "$OUT/run_metrics_${END}_d32_ms1M_lrep10.json" || true
  fi
  echo "=== done $END ==="
  START=$END
done
echo "REACHED $END_TARGET"
