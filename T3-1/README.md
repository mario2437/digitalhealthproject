# T3-1 (v2) 전이 예측 모델은 무엇을 보고 맞히는가 — 수집 흔적을 이용한 지름길 학습 점검

KSC2026 [학부생논문] 심사용 초안 + 분석 코드 + 사전 등록 SAP.
핵심 질문: MSK-MET 전이 장기 예측 모델이 생물학적 신호 대신 **검사 패널 버전·검사 시기(환자 등록 순번)·검체 품질** 같은
데이터 수집 흔적을 학습하는가? 흔적을 통제(제거·층화·단일 패널·시기 분할)하면 성능과 SHAP 설명은 어떻게 달라지는가?
(v1 주제 "유전체가 전이 장기를 말해주는가"에서 바꾼 것. T3-2의 NSCLC 전이 동시발생 네트워크와는 겹치지 않게 함.)

## 파일

| 경로 | 내용 |
|---|---|
| `code/SAP_prespecified.md` | **사전 등록 분석 계획서** (질문, 코호트, 흔적 변수 정의, H1~H4 성공 기준 숫자, 날짜). 결과를 본 뒤 기준을 바꾸지 말 것 |
| `draft/초안_심사용.md` / `.docx` | 심사용 초안. 요약·서론·관련 연구(표 1)·방법은 완성, 결과·결론 숫자는 `__` 빈칸, 표 2는 틀만 |
| `draft/figures/` | 분석 후 `fig1_artifact_signal.png`, `fig2_mitigation.png`를 여기로 복사하면 docx에 들어감 |
| `refs/refs.md` | 참고문헌 13편(+예비 5편), DOI/PMID, 초록 기준 메모, 라벨 정의 원문 확인 |
| `code/msk_features.py` | v1 재사용 모듈 (MSK-MET 읽기, 라벨, 341 공통 유전자 변이 행렬) — 수정 없음 |
| `code/t31_common.py` | 경로·시드(20260914)·LightGBM 고정 파라미터·5-겹 CV·(층화) AUROC·쌍체 부트스트랩·특징 묶음 조립 |
| `code/01_cohort.py` | 코호트 흐름, 흔적 변수 기술 통계(ID 접미사-패널 일치, 등록 순번-패널 Spearman, 십분위별 패널 구성) |
| `code/02_features.py` | 특징 묶음 T(원발암)·A(흔적)·FU(추적, 보조 전용)·clinical·genomic, 라벨(5장기 + HIGH_BURDEN) |
| `code/03_artifact_signal.py` | H1: A, T, T+A, B, B+A OOF AUROC, ΔAUROC(일반/패널 층화), 민감도 코호트 |
| `code/04_recording_intensity.py` | H2: 포아송 회귀 RR (패널·연속 흔적, 원발암 보정, 추적 보정) |
| `code/05_era_shift.py` | H3: 시간 분할(등록 순번 70/30), 패널 간 교차 분할, 일반화 격차 차이, O/E |
| `code/06_shap_shift.py` | H4: TreeSHAP 순위, 흔적 최고 순위, 유전자 상위 10개 Jaccard |
| `code/07_mitigation.py` | 완화 전략 M0~M3 별 흔적 이득 Δ(B+A−B)와 Δ(B−T) |
| `code/08_figures.py` | 그림 1, 2 |
| `code/run_all.sh` | 전체 실행 |

## 실행

```bash
cd code
bash run_all.sh                                        # 결과 → ../results, 그림 → ../figures
RESULTS_DIR=/경로/results FIGURES_DIR=/경로/figures bash run_all.sh
```
- Python: `dh` conda 환경의 `python` (다르면 `PYTHON=... bash run_all.sh`), 데이터: 저장소의 `data/msk_met_2021/` (읽기 전용, `MSK_MET_DIR`로 변경 가능)
- 전체 약 10분 내외(부트스트랩이 대부분; 빠른 확인은 `N_BOOT=200`). n_jobs ≤ 16. 시드 고정.
- **과목 규칙**: `SAP_prespecified.md`를 팀이 검토·확정(날짜 갱신)하기 전에는 `03~08`을 돌리지 마세요. `01`·`02`는 기술 통계와 특징만 만듭니다.
- 개별 실행 예: `cd code && RESULTS_DIR=../results python 04_recording_intensity.py` (번호 순서; 07은 03·05 결과 필요)

결과 파일 → 초안 빈칸: 4.1 `h1_perf_primary.tsv`·`h1_delta_primary.tsv`·`h1_summary.json`·`h1_delta_sensitivity.tsv` /
4.2·표 2 `h2_rate_ratios.tsv`·`h2_count_by_panel.tsv` / 4.3 `h3_gap_diff.tsv`·`h3_perf.tsv`·`m_mitigation.tsv`·`m_summary.json` /
4.4 `h4_shap_summary.tsv` / 3.2 `cohort_flow.json`(Spearman).

## 학생이 정해야 할 것 (SAP에 반영 후 날짜 갱신)

1. **성공 기준 숫자**: SAP 초안의 H1(ΔAUROC≥0.02, 2/5), H2(RR≥1.10), H3(격차 차이≥0.02, 2/5), H4(SHAP 순위≤10, 2/5)를 그대로 쓸지, 근거와 함께 조정할지.
2. **흔적 변수 목록**: 환자 ID 순번·검체 순서·순도·커버리지·수술-시퀀싱 기간을 모두 넣을지. 순도·커버리지는 "검체 품질"이면서 생물학(종양 세포 비율)과도 얽혀 있음.
3. **추적 기간(OS_MONTHS, OS_STATUS)**: 전이가 생존에 영향을 주므로 예측 특징에서 뺐고 H2 보정에만 씀. 보정 결과를 "추적 기간이 설명한다"고 해석할 때 과보정 가능성을 어떻게 서술할지.
4. **시간 분할 비율**(70/30)과 기준 변수(환자 ID 순번 vs 패널).
5. **HIGH_BURDEN 임계값** 3곳을 유지할지.
6. **TMB의 위치**: TMB는 패널 크기에 따라 계산이 달라지는 흔적 성격도 있음 — B에 둘지 A로 옮길지.
7. **그림 2개 선택**과 표 2 구성, 점검 목록(체크리스트)을 결론에 둘지 별도 상자로 둘지.

## 데이터 주의사항

- 라벨은 진단 시점 전이가 아니라 병리 보고서 + ICD 청구 코드로 **전체 임상 경과 동안** 기록된 전이(Nguyen 2022; 민감도 중앙값 77%). cBioPortal의 "at Diagnosis" 표기는 부정확.
- 샘플 ID 형식 `P-00XXXXX-T0n-IMk`: 접미사 IM3/IM5/IM6가 패널 341/410/468과 1:1 대응(=패널 원-핫과 중복이라 따로 넣지 않음). 환자 ID 숫자는 날짜가 아닌 등록 순번.
- 환자당 1검체이며 원 논문의 선택 규칙(FACETS QC > 순도 > 커버리지 > 최신 패널) 자체가 순도·커버리지·패널 분포에 영향을 줌.
- `AGE_AT_SURGERY`가 시퀀싱한 그 검체의 수술인지는 데이터로 확인할 수 없음. 음수 기간은 없었음. 결측은 LightGBM이 처리.
- 라벨 누출 변수(MET_COUNT, MET_SITE_COUNT, METASTATIC_SITE, AGE_AT_EVIDENCE_OF_METS)는 특징으로 쓰지 않음.
- 기타 유전체 처리 규칙(TERT 프로모터 포함, CNA ±2, PAK7→PAK5)은 v1과 같음(`msk_features.py` 주석).
