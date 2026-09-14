#!/usr/bin/env bash
# T3-1 (v2) 전체 파이프라인. 사용법:
#   bash run_all.sh                                    # 기본: ../results, ../figures
#   RESULTS_DIR=/경로/results FIGURES_DIR=/경로/figures bash run_all.sh
# 선택 환경변수: PYTHON (기본 dh 환경), MSK_MET_DIR (데이터 폴더), N_BOOT (부트스트랩 반복, 기본 2000)
# 과목 규칙: SAP_prespecified.md 를 확정(날짜 기록)하기 전에는 03~08 을 돌리지 말 것.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export RESULTS_DIR=${RESULTS_DIR:-../results}
export FIGURES_DIR=${FIGURES_DIR:-../figures}
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"
for s in 01_cohort.py 02_features.py 03_artifact_signal.py 04_recording_intensity.py 05_era_shift.py \
         06_shap_shift.py 07_mitigation.py 08_figures.py; do
  echo "=== $s ($(date +%T))"
  "$PY" "$s" 2>&1 | grep -v -i "warn" | tee "$RESULTS_DIR/log_${s%.py}.txt"
done
echo "done ($(date +%T))"
