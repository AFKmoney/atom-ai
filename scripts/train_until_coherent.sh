#!/usr/bin/env bash
# LR-decay ms1M until chat looks like a short FR sentence (heuristic), or END_TARGET.
set -euo pipefail
cd /workspace/exports/atom-ai
export PYTHONPATH=.
PY=.venv/bin/python
OUT=checkpoints/byte_tick
BASE=496000
END_TARGET=${END_TARGET:-5000000}
# --- branch parametrization (defaults reproduce the original lrd run exactly) ---
# SUFFIX     checkpoint/run_metrics/train-log suffix: atom_native_step_<N>_d32_ms1M_${SUFFIX}.pt
# LR / SLR   --learning-rate / --surface-learning-rate
# START_CKPT optional checkpoint (any suffix) used as --resume ONLY for the first chunk,
#            when no ${SUFFIX} checkpoint exists yet at START. START defaults to its step.
# LOG_TAG    tag for gen logs / coherence log / COHERENT_STEP file ("" for lrd, "_<SUFFIX minus lrd_>" otherwise)
SUFFIX=${SUFFIX:-lrd}
LR=${LR:-3e-5}
SLR=${SLR:-7.2e-5}
START_CKPT=${START_CKPT:-}
if [ -z "${LOG_TAG+x}" ]; then
  if [ "$SUFFIX" = "lrd" ]; then LOG_TAG=""; else LOG_TAG="_${SUFFIX#lrd_}"; fi
fi
TAG="d32_ms1M_${SUFFIX}"
START=${1:-}
if [ -z "$START" ]; then
  START=$(ls "$OUT"/atom_native_step_*_${TAG}.pt 2>/dev/null | sed -n 's/.*step_\([0-9]*\)_.*/\1/p' | sort -n | tail -1 || true)
fi
if [ -z "$START" ] && [ -n "$START_CKPT" ]; then
  START=$(basename "$START_CKPT" | sed -n 's/.*step_\([0-9]*\)_.*/\1/p')
fi
if [ -z "$START" ]; then echo "cannot determine START (no ${TAG} ckpt and no START_CKPT)"; exit 1; fi
COHERENCE_LOG=${COHERENCE_LOG:-logs/coherence_watch${LOG_TAG}.log}
COHERENT_STEP_FILE=${COHERENT_STEP_FILE:-logs/COHERENT_STEP${LOG_TAG}.txt}
mkdir -p logs
echo "=== $(date -Is) train_until_coherent SUFFIX=$SUFFIX LR=$LR SLR=$SLR START=$START END_TARGET=$END_TARGET START_CKPT=${START_CKPT:-none} LOG_TAG=${LOG_TAG:-none} ===" | tee -a "$COHERENCE_LOG"

is_coherent() {
  # stdin: generated text. True if looks like short FR phrase, not idi/irra attractor.
  python3 - <<'PY'
import sys,re
t=sys.stdin.read().strip()
low=t.lower()
# fail attractors
if any(x in low for x in ['idi','irra','ilisateur','tuu  tuu','::::']):
    sys.exit(1)
# need letters + a space (multi-word) or punctuation sentence
letters=sum(c.isalpha() for c in t)
if letters < 8:
    sys.exit(1)
words=re.findall(r"[A-Za-zÀ-ÿ']{2,}", t)
if len(words) < 3:
    sys.exit(1)
# reject if >40% spaces
if t.count(' ') > max(1, len(t)//2):
    sys.exit(1)
sys.exit(0)
PY
}

while [ "$START" -lt "$END_TARGET" ]; do
  END=$((START + 8000))
  [ "$END" -gt "$END_TARGET" ] && END=$END_TARGET
  STEPS=$((END - START))
  SKIP=$((START - BASE)); [ "$SKIP" -lt 0 ] && SKIP=0
  RESUME="$OUT/atom_native_step_${START}_${TAG}.pt"
  if [ ! -f "$RESUME" ] && [ -n "$START_CKPT" ] && [ "$(basename "$START_CKPT" | sed -n 's/.*step_\([0-9]*\)_.*/\1/p')" = "$START" ]; then
    RESUME="$START_CKPT"
  fi
  CKPT_NAME="atom_native_step_${END}_${TAG}.pt"
  LOG="logs/train_d32_${START}_${END}_ms1M_${SUFFIX}.log"
  if [ ! -f "$RESUME" ]; then echo "MISSING $RESUME"; exit 1; fi
  if [ ! -f "$OUT/$CKPT_NAME" ]; then
    echo "=== $(date -Is) $START->$END skip=$SKIP ===" | tee -a "$COHERENCE_LOG"
    $PY -u tools/run_atom_native.py \
      --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 \
      --stream-skip-packets "$SKIP" --resume "$RESUME" --output-dir "$OUT" \
      --checkpoint-name "$CKPT_NAME" --steps "$STEPS" \
      --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 \
      --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout \
      --efference-every 10 --learning-rate "$LR" --surface-learning-rate "$SLR" --seed 20260913 \
      --atom-flush-every 64 --episode-length 8000 --episode-reset --slow-every 1 \
      --field-loss-weight 0.05 --field-contrast-weight 0 --allow-parallel-train \
      > "$LOG" 2>&1
    [ -f "$OUT/run_metrics.json" ] && cp "$OUT/run_metrics.json" "$OUT/run_metrics_${END}_${TAG}.json" || true
  else
    echo "skip exists $CKPT_NAME" | tee -a "$COHERENCE_LOG"
  fi

  # coherence probe every tip (cold chat, 3 prompts)
  CKPT="$OUT/$CKPT_NAME"
  GENLOG="logs/gen_coherent_${END}${LOG_TAG}.txt"
  : > "$GENLOG"
  HIT=0
  for p in "Bonjour, comment ça va ?" "Qui es-tu ?" "Il était une fois"; do
    RESP=$($PY tools/chat_atom_native.py --checkpoint "$CKPT" --prompt "$p" --no-role-prime \
      --max-packets 96 --max-length 100 --temperature 0.8 --top-k 12 2>/dev/null | sed -n 's/^Response: //p' | head -1)
    echo "PROMPT: $p" >> "$GENLOG"
    echo "RESP: $RESP" >> "$GENLOG"
    if printf '%s' "$RESP" | is_coherent; then
      HIT=$((HIT+1))
      echo "COHERENT_CANDIDATE @$END :: $RESP" | tee -a "$COHERENCE_LOG"
    fi
  done
  # val line
  VAL=$(python3 -c "import json;d=json.load(open('$OUT/run_metrics_${END}_${TAG}.json'));print(d.get('validation',{}).get('loss'))" 2>/dev/null || echo '?')
  echo "tip=$END val=$VAL coherent_hits=$HIT/3" | tee -a "$COHERENCE_LOG"

  if [ "$HIT" -ge 2 ]; then
    echo "COHERENT_REACHED step=$END hits=$HIT" | tee -a "$COHERENCE_LOG"
    echo "$END" > "$COHERENT_STEP_FILE"
    exit 0
  fi
  START=$END
done
echo "HIT_END_TARGET $END_TARGET without coherence" | tee -a "$COHERENCE_LOG"
exit 2
