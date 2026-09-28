#!/usr/bin/env bash
# run_yield.sh NAME LOGFILE CMD...  — run CMD single-threaded in its own process
# group, started STOPPED; yield_to_live.py CONTs it only while the live b4828
# chunk is in its single-threaded stream-skip phase (or between chunks).
set -u
cd /workspace/exports/atom-ai
NAME=$1; LOG=$2; shift 2
export PYTHONPATH=${PYTHONPATH:-.} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PIDF=logs/fieldsat/${NAME}.pgid
setsid bash -c 'kill -STOP $$; exec "$@"' _ "$@" > "$LOG" 2>&1 &
PG=$!
echo $PG > "$PIDF"
sleep 0.2
.venv/bin/python docs/artifacts/field_saturation/yield_to_live.py "$PIDF" &
YP=$!
wait $PG; RC=$?
wait $YP
echo "EXIT $RC" >> "$LOG"
