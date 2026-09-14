#!/usr/bin/env bash
# T3-2(v2) 전체 파이프라인 실행.
# 사용법: bash run_all.sh [결과폴더] [그림폴더]
#   기본값: ../results  ../figures  (code/ 기준)
# 데이터 경로를 바꾸려면: T32_DATA_DIR=/경로 bash run_all.sh
# 과목 규칙: SAP_prespecified.md 를 확정(날짜 기록)하기 전에는 02 이후를 돌리지 마세요.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export T32_RESULTS_DIR=$(realpath -m "${1:-../results}")
export T32_FIGURES_DIR=$(realpath -m "${2:-../figures}")
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTHONWARNINGS=ignore
mkdir -p "$T32_RESULTS_DIR" "$T32_FIGURES_DIR"
for s in 01_cohort 02_cooccurrence 03_null_ising 04_rules 05_genotype 06_modules 07_figures 08_summary; do
  echo "===== $s ====="
  "$PY" "$s.py"
done 2>&1 | tee "$T32_RESULTS_DIR/run_log.txt"
echo "완료: 결과=$T32_RESULTS_DIR 그림=$T32_FIGURES_DIR"
