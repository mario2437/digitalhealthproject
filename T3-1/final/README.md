# T3-1 KSC2026 최종본 재현 자료

**수집 흔적에 기대는 전이 예측 모델의 점검과 완화: MSK-MET 임상 유전체 데이터의 시간 분할 평가**
*Auditing and Mitigating Shortcut Learning from Data-Collection Artifacts in Metastasis Prediction: A Temporal Evaluation on MSK-MET*
(KSC2026 한국정보과학회 학부생논문)

이 폴더는 최종 원고의 수치·표·그림을 만든 코드와 산출물을 모은 것입니다. 원고 파일은 포함하지 않습니다.
분석은 두 단계로 나뉩니다.

1. **v2 진단** — 사전 등록 기준 H1–H4. 코드: [`../code/`](../code) (`SAP_prespecified.md`, `01`–`08`)
2. **심화(해결 전략·평가 방식·중요도 대조)** — 사전 등록 부록 HD1–HD8. 코드: [`code/`](code) (`d1`–`d5`), 계획: [`SAP_addendum.md`](SAP_addendum.md)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `SAP_addendum.md` | 심화 분석 사전 계획. 2026-09-15 10:43:02 KST 확정, sha256 `b84bae03…4761f` (`results/deep/SAP_addendum.sha256.txt`). 원문 그대로 두었기 때문에 본문의 경로(`papers_v2/…`, `deep/…`)는 작성 당시 서버 경로입니다 — 이 저장소에서는 `T3-1/code/`, `T3-1/final/`에 해당합니다. |
| `code/deep_common.py` | v2 공통 모듈(`../code/t31_common.py`)을 재사용하고 입력·출력 경로를 이 폴더로 돌림 |
| `code/d1_strategies.py` | 해결 전략 S0–S7 × 4개 시점 분할(c50·c60·c70·c80) × 장기, 내부 CV(E0–E3)·시험 성능·보정 |
| `code/d12_evaluate.py` | 사전 규칙 R1–R3, HD1–HD6 판정, 내부 평가 방식 타당성(D2), ΔAUROC 부트스트랩 CI |
| `code/d3_importance.py` | 영(null) 특징 대조 SHAP 순위, 묶음 퍼뮤테이션(CV vs 시간 분할 시험), HD7·HD8 |
| `code/d4_selection_followup.py` | 기록 장기 수 포아송 회귀: 추적 기간·시퀀싱 시점 전이 확인(선택 대리) 순차 보정 |
| `code/d5_posthoc_figures.py` | 사후 요약(20개 조건 평균 이득·부풀림)과 그림 1·2 |
| `code/run_all_deep.sh` | 심화 전체 실행 |
| `results/v2/` | v2 산출물 전체 (코호트 ID, 특징 행렬 `features_*.pkl`, OOF·시험 예측값 포함) |
| `results/deep/` | 심화 산출물 전체 (`d1_test_pred_c70.tsv.gz` 시험 예측값 포함) |
| `figures/` | 원고 그림 1 = `fig_d1_strategies.png`, 그림 2 = `fig_d3_importance.png` (v2 그림 `fig1_artifact_signal.png`, `fig2_mitigation.png`는 참고용) |

## 원고 ↔ 파일 대응

| 원고 위치 | 내용 | 파일 |
|---|---|---|
| 3.1 | 원발 검체 11,741건, 원발암 27종, 패널 구성, 등록 순번–패널 ρ=0.81 | `results/v2/cohort_flow.json` |
| 4.1 | 흔적 단독 AUROC 0.57–0.62, 흔적 추가 ΔAUROC(H1) | `results/v2/h1_perf_primary.tsv`, `h1_delta_primary.tsv`, `h1_summary.json` |
| 4.1 | 기록 장기 수 비 1.40(H2) → 보정 후 1.03 (추적 76% + 선택 15%) | `results/v2/h2_rate_ratios.tsv`, `results/deep/d4_rr.tsv`, `d4_summary.json` |
| 4.1 | 시기 분할 격차(H3), SHAP 순위(H4) | `results/v2/h3_*`, `h4_*` |
| 표 2, 4.2 | 전략별 내부 CV·시험 AUROC·Δ·E/O(전체/추적 완료) | `results/deep/d1_strategy_table_c70.tsv`, `d1_delta_vs_S1_c70.tsv`, `d1_metrics.tsv` |
| 표 2, 4.2 | 20개 조건 평균 Δ와 ≥0 개수, 내부 부풀림 (사후분석) | `results/deep/d5_posthoc_strategy_summary.tsv`, `d5_posthoc_summary.json` |
| 4.2 | HD1–HD4 판정, 사전 규칙 권고 전략 S7 | `results/deep/d12_summary.json` |
| 그림 1 | 전략별 내부 이득 vs 시험 이득 | `figures/fig_d1_strategies.png` |
| 4.3 | 내부 평가 방식(E0–E3)의 Δ_A 오차·일치율, HD5·HD6 | `results/deep/d2_internal_validity.tsv`, `d12_summary.json` ("D2") |
| 4.4 | 영 특징 SHAP 11–18위, 등록 순번 퍼뮤테이션, HD7·HD8 | `results/deep/d3_shap_ranks.tsv`, `d3_perm_cv.tsv`, `d3_perm_external.tsv`, `d3_summary.json` |
| 그림 2 | 특징 묶음별 퍼뮤테이션 ΔAUROC | `figures/fig_d3_importance.png` |

판정 전체와 수치 해설은 각 `log_*.txt`에 있습니다.

## 재현

```bash
conda env create -f environment.yml && conda activate dh   # 저장소 루트에서
bash T3-1/code/run_all.sh                 # v2: 결과 → T3-1/results, 그림 → T3-1/figures (약 16분)
bash T3-1/final/code/run_all_deep.sh      # 심화: results/v2/features_primary.pkl 을 읽어 → results/deep, figures (약 10분)
```

- 데이터는 저장소의 `data/msk_met_2021/`를 그대로 씁니다.
- 시드 20260914 고정. 병렬 프로세스 수는 `MAX_JOBS`(심화, 기본 16)로 바꿀 수 있고 결과는 프로세스 수와 무관합니다.
- 다른 폴더에 다시 돌려 비교하려면 `RESULTS_DIR=… FIGURES_DIR=… bash T3-1/code/run_all.sh`, `DEEP_RESULTS=… bash T3-1/final/code/run_all_deep.sh`.

## 데이터·라이선스

MSK-MET (Nguyen B, et al. *Cell* 2022;185(3):563–575; cBioPortal `msk_met_2021`)에서 파생한 자료입니다.
`results/`의 환자 단위 파일을 포함한 파생 자료는 원자료와 같이 [ODbL 1.0](../../data/msk_met_2021/LICENSE)을 따릅니다.
전이 라벨은 진단 시점이 아니라 전체 경과 중 기록된 전이이며(Nguyen 2022), 이 점은 원고의 한계에 적었습니다.
