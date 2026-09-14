#!/usr/bin/env bash
# T2(v2) 파이프라인: 한국어 식품명 기반 NOVA 자동 분류
#
# 사용법
#   bash run_all.sh prepare     # 1-3단계: 원시자료 추출 → 식품코드 표 → 정답셋 표본·배정 템플릿
#   (SAP_prespecified.md 확정 → 두 배정자가 gold_rater_A/B.csv를 독립 배정)
#   bash run_all.sh kappa       # 4단계: 배정자 간 kappa → 합의 초안 → (합의 후) results/gold_labels.csv 저장
#   bash run_all.sh classify    # 5-7단계: 규칙 → 텍스트 ML → LLM (GPU 필요)
#   bash run_all.sh evaluate    # 8-10단계: 성능·UPF% 영향·성공 기준·그림
#   bash run_all.sh all         # classify + evaluate (정답셋이 이미 있을 때)
#
# 환경변수: T2_RESULTS_DIR, T2_FIGURES_DIR, T2_GOLD, T2_LLM_MODELS, CUDA_VISIBLE_DEVICES, PY
#   예) CUDA_VISIBLE_DEVICES=0 T2_RESULTS_DIR=/tmp/r T2_FIGURES_DIR=/tmp/f bash run_all.sh all
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
STEP=${1:-all}

need_gold() {
  local g=${T2_GOLD:-${T2_RESULTS_DIR:-../results}/gold_labels.csv}
  [ -f "$g" ] || { echo "정답셋 파일이 없습니다: $g (4단계 합의 후 저장)"; exit 1; }
}
classify() {
  need_gold
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "CUDA_VISIBLE_DEVICES를 지정하세요 (nvidia-smi로 빈 GPU 확인)"; exit 1; }
  $PY 05_rules.py && $PY 06_textml.py && $PY 07_llm.py --scope gold
}
evaluate() { need_gold; $PY 08_evaluate.py && $PY 09_upf_impact.py && $PY 10_figures.py; }

case "$STEP" in
  prepare)  $PY 01_extract.py && $PY 02_foodcode_table.py && $PY 03_sample_gold.py ;;
  kappa)    $PY 04_kappa_consensus.py ;;
  classify) classify ;;
  evaluate) evaluate ;;
  all)      classify && evaluate ;;
  *) echo "알 수 없는 단계: $STEP (prepare|kappa|classify|evaluate|all)"; exit 1 ;;
esac
