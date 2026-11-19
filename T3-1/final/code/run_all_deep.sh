#!/usr/bin/env bash
# T3-1 심화 분석 파이프라인 (v2 결과 features_primary.pkl 재사용, v2 산출물은 건드리지 않음)
#   bash run_all_deep.sh                 # 결과: ../results/deep, 그림: ../figures
#   DEEP_RESULTS=/다른/경로 bash run_all_deep.sh   # 재현성 점검용 (d12·d5는 이 경로의 d1·d3 결과를 읽음)
# 소요: d1 약 4분(16 프로세스), d3 약 5분, d4 1분 미만.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
OUT=${DEEP_RESULTS:-../results/deep}
mkdir -p "$OUT"
for s in d1_strategies.py d12_evaluate.py d3_importance.py d4_selection_followup.py d5_posthoc_figures.py; do
  echo "=== $s ($(date +%T))"
  "$PY" "$s" 2>&1 | grep -v -i "warn" | tee "$OUT/log_${s%.py}.txt"
done
echo "done ($(date +%T))"
