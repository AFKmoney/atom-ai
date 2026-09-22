#!/usr/bin/env bash
set -euo pipefail
cd /workspace/exports/atom-ai
export PYTHONPATH=.
PY=.venv/bin/python
OUT=checkpoints/byte_tick
BASE=496000
END_TARGET=2000000
START=${1:-880000}
RESUME0="$OUT/atom_native_step_880000_d32_ms1M_lrdecay.pt"
FIRST=1
while [ "$START" -lt "$END_TARGET" ]; do
  END=$((START + 8000))
  [ "$END" -gt "$END_TARGET" ] && END=$END_TARGET
  STEPS=$((END - START))
  SKIP=$((START - BASE)); [ "$SKIP" -lt 0 ] && SKIP=0
  if [ "$FIRST" = 1 ] && [ "$START" = 880000 ]; then
    RESUME="$RESUME0"; FIRST=0
  else
    RESUME="$OUT/atom_native_step_${START}_d32_ms1M_lrd.pt"
    # allow resume from plain tip if lrd missing (chain)
    if [ ! -f "$RESUME" ] && [ -f "$OUT/atom_native_step_${START}_d32_ms1M.pt" ]; then
      RESUME="$OUT/atom_native_step_${START}_d32_ms1M.pt"
    fi
    if [ ! -f "$RESUME" ] && [ -f "$RESUME0" ] && [ "$START" = 880000 ]; then
      RESUME="$RESUME0"
    fi
  fi
  CKPT_NAME="atom_native_step_${END}_d32_ms1M_lrd.pt"
  LOG="logs/train_d32_${START}_${END}_ms1M_lrd.log"
  if [ ! -f "$RESUME" ]; then echo "MISSING $RESUME"; exit 1; fi
  if [ -f "$OUT/$CKPT_NAME" ]; then echo "skip $CKPT_NAME"; START=$END; continue; fi
  echo "=== $(date -Is) $START->$END skip=$SKIP LR-decay ep8000 ==="
  $PY -u tools/run_atom_native.py \
    --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 \
    --stream-skip-packets "$SKIP" --resume "$RESUME" --output-dir "$OUT" \
    --checkpoint-name "$CKPT_NAME" --steps "$STEPS" \
    --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 \
    --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout \
    --efference-every 10 --learning-rate 3e-5 --surface-learning-rate 7.2e-5 --seed 20260913 \
    --atom-flush-every 64 --episode-length 8000 --episode-reset --slow-every 1 \
    --field-loss-weight 0.05 --field-contrast-weight 0 --allow-parallel-train \
    > "$LOG" 2>&1
  [ -f "$OUT/run_metrics.json" ] && cp "$OUT/run_metrics.json" "$OUT/run_metrics_${END}_d32_ms1M_lrd.json" || true
  echo "=== done $END ==="
  START=$END
done
echo REACHED
