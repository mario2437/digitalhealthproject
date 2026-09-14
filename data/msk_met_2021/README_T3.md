# MSK-MET (msk_met_2021) — T3 암 전이 장기 예측

출처: cBioPortal `msk_met_2021` — MSK MetTropism (MSK, Cell 2021), Nguyen et al.
받은 날짜: 2026-09-10 / 받은 방법: datahub git-lfs + cBioPortal REST API

## 파일

| 파일 | 크기 | 내용 |
|---|---|---|
| data_clinical_sample.txt | 6.8M | 샘플 25,775건. **전이 장기 라벨 21종 (DMETS_DX_*)**, PRIMARY_SITE, CANCER_TYPE, FGA, TMB, MSI, GENE_PANEL |
| data_clinical_patient.txt | 1.6M | 환자 단위 임상 (생존 등) |
| data_mutations.txt | 27M | 변이 227,931건 (sampleId, hugoGeneSymbol, mutationType, proteinChange 등) |
| data_cna.txt | 28M | 유전자 단위 copy number |
| data_gene_panel_matrix.txt | 1.2M | 샘플별 사용 패널 (IMPACT341 / 410 / 468) |
| data_sv.txt | 2.0M | 구조변이 |

## 바로 알아야 할 것

- **라벨 인코딩은 0/1이 아니라 `Yes`/`No`** 입니다.
- `IS_DIST_MET_MAPPED == TRUE` 인 **21,546건만** 전이 부위가 매핑되어 있습니다. 나머지 4,229건은
  MET_SITE_COUNT=0 이며, 이걸 "전이 없음"으로 볼지 "정보 없음"으로 볼지 SAP에 명시해야 합니다.
- 5개 목표 장기 유병률 (매핑된 21,546건 기준):
  폐 36.8% / 간 40.8% / 뼈 33.4% / 뇌 13.6% / 복강내 28.6%
  → 뇌가 가장 드물지만 2,921건이므로 분석 가능합니다. 클래스 불균형 처리 + AUPRC 병기.
- SAMPLE_TYPE: Primary 15,632 / Metastasis 10,143.
  **전이 검체에서 뽑은 변이로 전이 부위를 예측하면 순환 논리입니다.** Primary만 쓰거나,
  샘플 타입을 보정하고 층화 분석하세요. 심사에서 반드시 나올 질문입니다.
- **패널이 3종(341/410/468 유전자)으로 섞여 있습니다.** 패널마다 검사한 유전자가 달라서,
  "변이 없음"이 진짜 없음인지 안 봤음인지 구분되지 않습니다. 341개 공통 유전자로 제한하거나
  패널을 공변량으로 넣으세요. data_gene_panel_matrix.txt 가 그 근거입니다.
- FGA·TMB는 25,775건 전체에 값이 있습니다.

## 주의
data_mutations.txt 는 datahub의 git-lfs 객체가 서버에 없어서(404), cBioPortal REST API로
직접 받아 재구성한 것입니다. 건수(227,931)는 API의 total-count와 일치합니다.
다만 컬럼 구성은 원본 MAF와 다릅니다 — 원본 MAF 컬럼이 필요하면 cBioPortal 웹에서 직접 받으세요.

## 유전자 패널 정의 (2026-09-13 추가)
`gene_panels/` — cBioPortal REST API(`/api/gene-panels/{id}`)로 받음 (datahub 원본은 git-lfs라 받을 수 없음).
- IMPACT341 ⊂ IMPACT410 ⊂ IMPACT468 → 3개 패널 공통 유전자 = IMPACT341의 341개 (`common_genes.txt`)
- T3-2 사전 지정 후보(EGFR, KRAS, TP53, STK11, KEAP1, SMARCA4, CDKN2A, ERBB2, BRAF, MET, PIK3CA, RB1, ALK, ROS1, RET, NTRK1)는 모두 공통 유전자에 포함
