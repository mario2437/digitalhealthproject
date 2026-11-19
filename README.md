# 디지털헬스 프로젝트 (2026 가을) — 코드와 데이터

KSC2026(한국정보과학회) 학부생논문 준비용 팀별 분석 코드, 사전 분석계획서(SAP), 논문 초안 틀입니다.

| 폴더 | 주제 | 데이터 |
|---|---|---|
| [`T1/`](T1) | 조사연도가 바뀌면 예측모형은 왜 틀리는가 — 시간적 데이터셋 이동 분해 | KNHANES (직접 내려받기) |
| [`T2/`](T2) | 한국어 식품명 기반 NOVA 초가공식품 자동 분류 (로컬 LLM) | KNHANES (직접 내려받기) |
| [`T3-1/`](T3-1) | 전이 예측 모형의 지름길 학습(패널·시기 흔적) 점검 — 최종본 재현 자료 [`T3-1/final/`](T3-1/final) | MSK-MET (포함) |
| [`T3-2/`](T3-2) | NSCLC 전이 장기 동시발생 네트워크 | MSK-MET (포함) |
| [`common/`](common) | KSC 심사용 docx 변환(`md2docx.py`), 그림 스타일(`plotstyle.py`), 양식 | |
| [`data/`](data) | `msk_met_2021/` 포함, `knhanes/` 안내만 | |

각 팀 폴더의 `README.md`에 파일 설명, 실행 순서, 팀이 결정할 사항이 있습니다.

## 시작하기

```bash
git clone <이 저장소 URL>
cd digital-health-project
conda env create -f environment.yml      # T2는 environment-llm.yml 도 추가
conda activate dh
```

- **T3-1, T3-2**: 바로 실행할 수 있습니다. `bash T3-1/code/run_all.sh`
- **T1, T2**: 먼저 [`data/knhanes/README.md`](data/knhanes/README.md)대로 원시자료를 받아 넣으세요.
- 한글 그림에는 NanumGothic 글꼴이 필요합니다(`sudo apt install fonts-nanum` 또는 `~/.local/share/fonts`에 복사).

## 연구 진행 규칙

1. `code/SAP_prespecified.md`를 **결과를 보기 전에** 팀이 확정하고 시각을 기록합니다.
2. 결과를 본 뒤에는 성공 기준을 바꾸지 않습니다. 추가 분석은 "사후 분석"으로 표시합니다.
3. 기준을 충족하지 못한 결과도 그대로 보고합니다.

## 협업

- `main`에 바로 올리지 말고 팀 브랜치(예: `t1/sap-update`)에서 작업한 뒤 Pull Request를 여세요.
- **원시자료, 개인 단위 가공 파일, `results/`는 커밋하지 않습니다.** (예외: `T3-1/final/`은 최종본 재현용으로 결과까지 공개)

## 데이터 출처와 라이선스

- **MSK-MET** (`data/msk_met_2021/`): Nguyen B, et al. *Genomic characterization of metastatic patterns
  from prospective clinical sequencing of 25,000 patients.* Cell 2022;185(3):563–575.
  cBioPortal `msk_met_2021`. [ODC Open Database License (ODbL) 1.0](data/msk_met_2021/LICENSE) —
  이 자료를 가공해 공개할 때도 출처를 밝히고 ODbL로 공개해야 합니다.
- **KNHANES**: 질병관리청 국민건강영양조사. 원시자료 이용 규정에 따라 저장소에 포함하지 않습니다.
