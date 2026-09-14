#!/usr/bin/env bash
# T1(v2) 분석 전체 실행.
# 사용법:  bash run_all.sh [출력루트]
#   출력루트를 주면 <출력루트>/results, <출력루트>/figures 에 저장 (환경변수 T1_OUT 와 동일)
#   생략하면 이 폴더 기준 ../results, ../figures 에 저장
# 주의: 03 이후 스크립트는 결과변수를 사용합니다. SAP_prespecified.md 를 확정(시각 기록)한 뒤 실행하세요.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
if [ $# -ge 1 ]; then export T1_OUT="$1"; fi
# 병렬은 joblib(최대 16 프로세스)로 하므로 수치 라이브러리 스레드는 1로 고정
export PYTHONHASHSEED=2026 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
for s in 01_build_dataset.py 02_covariate_shift.py 03_temporal_validation.py 04_decomposition.py \
         05_mets.py 06_figures.py 07_criteria.py; do
  echo "=== $s"
  "$PY" "$s"
done
echo "완료: ${T1_OUT:-..}/results, ${T1_OUT:-..}/figures"
