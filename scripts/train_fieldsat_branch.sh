#!/usr/bin/env bash
# Resumable driver for the field-leak branch (docs/FIELD_SATURATION.md).
#
# Continues the fieldsat branch in 8k chunks with the exact live recipe from
# scripts/train_until_coherent.sh + --last-atom-dropout 0.5 + --field-leak 0.3.
# Each chunk saves checkpoints/byte_tick_fieldsat/atom_native_step_<END>_d32_ms1M_lrd_b4828_fieldsat.pt,
# so the run can be resumed or switched at any chunk boundary.  Re-running the
# script resumes from the newest existing branch checkpoint in [START_STEP, END_TARGET]
# (an interrupted chunk is simply re-run) and measures any checkpoint that has no
# row in docs/artifacts/field_saturation/branch_progress.tsv yet.
#
# Env:
#   THREADS=1     OMP/MKL threads for every python step (8 only when the live loop is stopped)
#   YIELD=1       1 = SIGSTOP this whole process group while the live b4828 chunk is in its
#                 multi-threaded phase (docs/artifacts/field_saturation/yield_to_live.py);
#                 requires launching via setsid (own process group). 0 = never pause.
#   START_STEP=4876000  END_TARGET=4916000  CHUNK=8000
#   LR=1.5e-5 SLR=3.6e-5 LEAK=0.3 LADROP=0.5
#
# Launch (background, own process group):
#   setsid nohup env THREADS=1 YIELD=1 bash scripts/train_fieldsat_branch.sh \
#       >> logs/fieldsat_branch_master.log 2>&1 < /dev/null &
# Stop cleanly (group may be SIGSTOPped, so CONT after TERM):
#   PG=$(cat logs/fieldsat/branch.pgid); kill -TERM -- -$PG; kill -CONT -- -$PG
set -u
cd /workspace/exports/atom-ai
THREADS=${THREADS:-1}
YIELD=${YIELD:-1}
START_STEP=${START_STEP:-4876000}
END_TARGET=${END_TARGET:-4916000}
CHUNK=${CHUNK:-8000}
LR=${LR:-1.5e-5}
SLR=${SLR:-3.6e-5}
LEAK=${LEAK:-0.3}
LADROP=${LADROP:-0.5}
BASE=496000
OUT=checkpoints/byte_tick_fieldsat
SUF=d32_ms1M_lrd_b4828_fieldsat
A=docs/artifacts/field_saturation
AB=$A/branch
TSV=$A/branch_progress.tsv
LOGD=logs/fieldsat/branch
RING=$OUT/val_ring_4876000.pt   # fixed 256-transition stream-ring val (same window as the probe table)
PY=.venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
mkdir -p "$OUT" "$AB" "$LOGD" logs/fieldsat
ts() { date -Is; }

exec 9>logs/fieldsat/branch.lock
if ! flock -n 9; then echo "$(ts) another train_fieldsat_branch.sh holds logs/fieldsat/branch.lock; exiting"; exit 1; fi

PG=$(ps -o pgid= -p $$ | tr -d ' ')
echo "$PG" > logs/fieldsat/branch.pgid
echo "=== $(ts) fieldsat branch driver pid=$$ pgid=$PG THREADS=$THREADS YIELD=$YIELD START_STEP=$START_STEP END_TARGET=$END_TARGET CHUNK=$CHUNK LR=$LR SLR=$SLR LEAK=$LEAK LADROP=$LADROP ==="
if [ "$YIELD" = "1" ]; then
  if [ "$PG" != "$$" ]; then
    echo "$(ts) YIELD=1 needs its own process group: launch with setsid (see header). exiting"; exit 1
  fi
  # Helper in its own session (so it never stops itself); pauses/resumes only our pgid.
  setsid "$PY" "$A/yield_to_live.py" logs/fieldsat/branch.pgid 9>&- \
    >> logs/fieldsat/branch_yield.err 2>&1 < /dev/null &
  echo "$(ts) yield helper pid=$! watching pgid=$PG"
fi

latest() {
  ls "$OUT"/atom_native_step_*_${SUF}.pt 2>/dev/null \
    | sed -n 's/.*step_\([0-9]*\)_.*/\1/p' | sort -n \
    | awk -v lo="$START_STEP" -v hi="$END_TARGET" '$1>=lo && $1<=hi' | tail -1
}

has_row() { [ -f "$TSV" ] && awk -F'\t' -v s="$1" 'NR>1 && $1==s {f=1} END {exit !f}' "$TSV"; }

measure() {
  local END=$1 CK="$OUT/atom_native_step_$1_${SUF}.pt" ML="$LOGD/measure_$1.log"
  echo "$(ts) measure $END start"
  "$PY" -u tools/eval_last_atom_ablation.py --end-steps 4876000 --ring-cache "$RING" \
    --checkpoints "$CK" --out "$AB/val_ablation_$END.json" >> "$ML" 2>&1
  "$PY" -u tools/diag_field_saturation.py --checkpoints "b$END=$CK" --episode-ticks 8000 \
    --prefixes 3000 --probe --out "$AB/diag_$END.json" >> "$ML" 2>&1
  local G="$AB/gen_coherent_$END.txt"; : > "$G"
  for p in "Bonjour, comment ça va ?" "Qui es-tu ?" "Il était une fois"; do
    RESP=$("$PY" tools/chat_atom_native.py --checkpoint "$CK" --prompt "$p" --no-role-prime \
      --max-packets 96 --max-length 100 --temperature 0.8 --top-k 12 2>/dev/null | sed -n 's/^Response: //p' | head -1)
    echo "PROMPT: $p" >> "$G"; echo "RESP: $RESP" >> "$G"
  done
  "$PY" - "$END" "$CK" "$TSV" "$AB" "$OUT/run_metrics_${END}_${SUF}.json" <<'PYEOF'
import json, os, sys
end, ck, tsv, ab, rm_path = sys.argv[1:6]
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
rm = load(rm_path); tr = rm.get("train", {}); va = rm.get("validation", {})
ev = load(f"{ab}/val_ablation_{end}.json")
ring = next(iter(ev.get("rings", {}).values()), {}).get("checkpoints", {}).get(ck, {})
dg = (load(f"{ab}/diag_{end}.json").get("rows") or [{}])[0]
fork = (dg.get("shared_prefix_forks") or {}).get("3000", {})
chats = []
try:
    for line in open(f"{ab}/gen_coherent_{end}.txt", encoding="utf-8"):
        if line.startswith("RESP: "):
            chats.append(" ".join(line[6:].split())[:80].replace("\t", " "))
except FileNotFoundError:
    pass
def f(x, n=4):
    return "" if x is None or x == "" else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
cols = [
    ("end_step", end),
    ("chunk_val_loss_own_ring", f(va.get("loss"))),
    ("val_loss_fixed_ring", f(ring.get("normal", {}).get("loss"))),
    ("val_byte_ce_fixed_ring", f(ring.get("normal", {}).get("byte_loss"))),
    ("field_alone_byte_ce", f(ring.get("la_zeroed", {}).get("byte_loss"))),
    ("a_hat_linear_next_byte_ce", f((dg.get("probe_next_byte_from_a_hat") or {}).get("test_ce"), 3)),
    ("train_frac_at_cap", f(tr.get("field_limited_fraction"), 3)),
    ("diag_frac_at_cap", f(dg.get("frac_at_cap"), 3)),
    ("fork_cos_alpha", f(fork.get("mean_pairwise_cos_alpha"), 3)),
    ("fork_cos_a_hat", f(fork.get("mean_pairwise_cos_a_hat"), 3)),
    ("train_steps_per_s", f(tr.get("transitions_per_second"), 1)),
    ("field_leak", f((rm.get("config") or {}).get("field_leak"), 2)),
    ("chat_bonjour", chats[0] if len(chats) > 0 else ""),
    ("chat_qui", chats[1] if len(chats) > 1 else ""),
    ("chat_il_etait", chats[2] if len(chats) > 2 else ""),
]
new = not os.path.exists(tsv)
with open(tsv, "a", encoding="utf-8") as fh:
    if new:
        fh.write("\t".join(c for c, _ in cols) + "\n")
    fh.write("\t".join(v for _, v in cols) + "\n")
print("TSV:", "\t".join(v for _, v in cols))
PYEOF
  echo "$(ts) measure $END done"
}

measure_missing() {
  for ck in $(ls "$OUT"/atom_native_step_*_${SUF}.pt 2>/dev/null | sed -n 's/.*step_\([0-9]*\)_.*/\1/p' | sort -n); do
    if [ "$ck" -ge "$START_STEP" ] && [ "$ck" -le "$END_TARGET" ] && ! has_row "$ck"; then measure "$ck"; fi
  done
}

CUR=$(latest)
if [ -z "$CUR" ]; then echo "$(ts) no branch checkpoint in [$START_STEP,$END_TARGET] under $OUT; exiting"; exit 1; fi
while [ "$CUR" -lt "$END_TARGET" ]; do
  END=$((CUR + CHUNK)); [ "$END" -gt "$END_TARGET" ] && END=$END_TARGET
  STEPS=$((END - CUR)); SKIP=$((CUR - BASE)); [ "$SKIP" -lt 0 ] && SKIP=0
  RESUME="$OUT/atom_native_step_${CUR}_${SUF}.pt"
  NAME="atom_native_step_${END}_${SUF}.pt"
  LOG="$LOGD/train_${CUR}_${END}_fieldsat.log"
  if [ ! -f "$OUT/$NAME" ]; then
    echo "$(ts) chunk $CUR->$END skip=$SKIP resume=$RESUME log=$LOG"
    T0=$(date +%s)
    "$PY" -u tools/run_atom_native.py \
      --stream --data-glob 'data/corpus_fr_medium*.txt' --loop-shards --chunk-bytes 65536 \
      --stream-skip-packets "$SKIP" --resume "$RESUME" --output-dir "$OUT" \
      --checkpoint-name "$NAME" --steps "$STEPS" \
      --d-model 32 --n-modes 32 --n-atoms-max 64 --max-span-bytes 1 \
      --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout \
      --efference-every 10 --learning-rate "$LR" --surface-learning-rate "$SLR" --seed 20260913 \
      --atom-flush-every 64 --episode-length 8000 --episode-reset --slow-every 1 \
      --field-loss-weight 0.05 --field-contrast-weight 0 --allow-parallel-train \
      --last-atom-dropout "$LADROP" --field-leak "$LEAK" \
      > "$LOG" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ] || [ ! -f "$OUT/$NAME" ]; then
      echo "$(ts) chunk $CUR->$END FAILED rc=$RC (see $LOG)"; exit 1
    fi
    cp "$OUT/run_metrics.json" "$OUT/run_metrics_${END}_${SUF}.json"
    cp "$OUT/run_metrics.json" "$AB/run_metrics_${END}_fieldsat.json"
    echo "$(ts) chunk $CUR->$END saved $NAME wall=$(( $(date +%s) - T0 ))s"
  else
    echo "$(ts) skip exists $NAME"
  fi
  measure_missing
  CUR=$END
done
measure_missing
echo "$(ts) DONE fieldsat branch at $CUR (END_TARGET=$END_TARGET)"
