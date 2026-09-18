#!/usr/bin/env bash
# Three L_ign mechanism smokes: ablate / prompt-bank / both. Docs+logs only push.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}"

SRC_CKPT="checkpoints/atom_native_lign_smoke/atom_native_src.pt"
DATA="data/corpus_train_chat.txt"
COMMON=(
  --data "$DATA"
  --resume "$SRC_CKPT"
  --steps 2500
  --d-model 64 --n-modes 64 --n-atoms-max 512
  --max-span-bytes 16 --episode-length 512 --no-episode-reset
  --atom-flush-every 256 --field-max-rms 3.0
  --learning-rate 0.0001 --surface-learning-rate 0.0003
  --energy-decay-min 0.45 --energy-decay-max 0.95
  --enable-merge --field-loss-weight 0.08
  --field-contrast-weight 0.45 --field-contrast-margin 0.45
  --field-ignorance-weight 0.08 --field-ignorance-margin 0.85
  --slow-every 4 --seed 20260917 --log-every 200
)

probe() {
  local ckpt="$1" out_txt="$2" report_json="$3"
  "$PY" -u tools/probe_field_persistence.py \
    --checkpoint "$ckpt" --skip-docs --report-json "$report_json" \
    2>&1 | tee "$out_txt"
}

extract_cos() {
  # prints: logits alpha from probe stdout
  local f="$1"
  rg -o 'mean off-diag cosine logits=[0-9.]+ alpha=[0-9.]+' "$f" | tail -1 \
    | sed -E 's/.*logits=([0-9.]+) alpha=([0-9.]+)/\1 \2/'
}

run_one() {
  local name="$1" out_dir="$2" art_dir="$3"; shift 3
  local extra=("$@")
  mkdir -p "$out_dir" "$art_dir"
  echo "===== BEGIN $name $(date -Is) ====="

  # BEFORE probe (src)
  probe "$SRC_CKPT" "$art_dir/probe_before.txt" "$art_dir/probe_before_report.json"
  read -r BEFORE_LOGITS BEFORE_ALPHA < <(extract_cos "$art_dir/probe_before.txt")

  local train_log="$art_dir/train_log.txt"
  echo "===== TRAIN $name =====" | tee "$train_log"
  "$PY" -u tools/run_atom_native.py \
    "${COMMON[@]}" \
    --output-dir "$out_dir" \
    "${extra[@]}" \
    2>&1 | tee -a "$train_log"

  # AFTER probe
  probe "$out_dir/atom_native.pt" "$art_dir/probe_after.txt" "$art_dir/probe_after_report.json"
  read -r AFTER_LOGITS AFTER_ALPHA < <(extract_cos "$art_dir/probe_after.txt")

  # Excerpt + metrics JSON
  {
    echo "# train log excerpt — $name"
    echo
    head -n 8 "$train_log"
    echo "..."
    rg -n 'step=|run |ign=|mph=|merges=' "$train_log" | head -n 40 || true
    echo "..."
    tail -n 30 "$train_log"
  } > "$art_dir/train_log_excerpt.txt"

  # Copy run_metrics if present
  if [[ -f "$out_dir/run_metrics.json" ]]; then
    cp "$out_dir/run_metrics.json" "$art_dir/run_metrics.json"
  fi

  # Summarize ign/mph/merges from train log
  local final_ign final_mph merges_any
  final_ign=$(rg -o 'ign=[0-9.]+' "$train_log" | tail -1 | cut -d= -f2 || echo nan)
  final_mph=$(rg -o 'mph=[0-9.]+' "$train_log" | tail -1 | cut -d= -f2 || echo nan)
  merges_any=$(rg -c 'merges=[1-9]' "$train_log" || true)
  merges_any=${merges_any:-0}

  cat > "$art_dir/series_row.json" << JSON
{
  "name": "$name",
  "before_logits_cos": ${BEFORE_LOGITS:-null},
  "after_logits_cos": ${AFTER_LOGITS:-null},
  "before_alpha_cos": ${BEFORE_ALPHA:-null},
  "after_alpha_cos": ${AFTER_ALPHA:-null},
  "final_ign": ${final_ign:-null},
  "final_mph": ${final_mph:-null},
  "merges_nonzero_log_lines": ${merges_any},
  "checkpoint": "$out_dir/atom_native.pt",
  "flags": $(printf '%s\n' "${extra[@]}" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read().split()))')
}
JSON
  echo "===== END $name logits $BEFORE_LOGITS -> $AFTER_LOGITS alpha $BEFORE_ALPHA -> $AFTER_ALPHA ====="
}

echo "SERIES START $(date -Is)" | tee logs/lign_series_driver.txt

run_one "ablate" \
  "checkpoints/atom_native_lign_ablate" \
  "docs/artifacts/lign_ablate_2k5" \
  --field-ignorance-ablate-shared --no-field-ignorance-prompt-bank \
  2>&1 | tee -a logs/lign_series_driver.txt

run_one "prompt_bank" \
  "checkpoints/atom_native_lign_prompt_bank" \
  "docs/artifacts/lign_prompt_bank_2k5" \
  --no-field-ignorance-ablate-shared --field-ignorance-prompt-bank \
  2>&1 | tee -a logs/lign_series_driver.txt

run_one "both" \
  "checkpoints/atom_native_lign_both" \
  "docs/artifacts/lign_both_2k5" \
  --field-ignorance-ablate-shared --field-ignorance-prompt-bank \
  2>&1 | tee -a logs/lign_series_driver.txt

echo "SERIES DONE $(date -Is)" | tee -a logs/lign_series_driver.txt
