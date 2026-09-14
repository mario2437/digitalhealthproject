"""01. 코호트 정의 + 수집 흔적 변수 기술 통계 (라벨 모델 없음).

주 분석: IS_DIST_MET_MAPPED == True & SAMPLE_TYPE == Primary (원발 불명/유전체 없음 제외)
민감도: 매핑된 Primary + Metastasis.
흔적 변수 확인: 환자 ID 번호(PID_NUM)가 패널 도입 순서(341→410→468)와 단조 관계인지.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import msk_features as mf
from t31_common import RESULTS

clin = mf.load_clinical()
assert clin.PATIENT_ID.is_unique, "환자당 샘플이 1개가 아님"
flow = {"전체 샘플(=환자)": len(clin)}

panel = pd.read_csv(mf.DATA_DIR / "data_gene_panel_matrix.txt", sep="\t")
cna_cols = pd.read_csv(mf.DATA_DIR / "data_cna.txt", sep="\t", nrows=0).columns[1:]
has_genomic = clin.SAMPLE_ID.isin(panel.SAMPLE_ID) & clin.SAMPLE_ID.isin(cna_cols)
unknown = clin.CANCER_TYPE.isna() | clin.CANCER_TYPE.str.contains("Unknown", case=False, na=True)

mapped = clin[clin.IS_DIST_MET_MAPPED]
flow["제외: 전이 부위 미매핑"] = int((~clin.IS_DIST_MET_MAPPED).sum())
flow["제외: Metastasis 검체(민감도에만)"] = int((mapped.SAMPLE_TYPE != "Primary").sum())
prim = mapped[mapped.SAMPLE_TYPE == "Primary"]
flow["제외: 원발 불명"] = int(unknown[prim.index].sum())
flow["제외: 유전체 프로파일 없음"] = int((~has_genomic[prim.index]).sum())
prim = prim[~unknown[prim.index] & has_genomic[prim.index]].copy()
sens = mapped[~unknown[mapped.index] & has_genomic[mapped.index]].copy()
flow["주 분석 코호트"] = len(prim)
flow["민감도 코호트"] = len(sens)
flow["원발암 종류 수"] = int(prim.CANCER_TYPE.nunique())
flow["패널"] = prim.GENE_PANEL.value_counts().to_dict()

# --- 흔적 변수 점검 (라벨 미사용) ---
pid = prim.SAMPLE_ID.str.extract(r"^P-(\d+)-", expand=False).astype(int)
suffix = prim.SAMPLE_ID.str.extract(r"-(IM\d)$", expand=False)
tnum = prim.SAMPLE_ID.str.extract(r"-T(\d+)-", expand=False).astype(int)
order = prim.GENE_PANEL.map({"IMPACT341": 0, "IMPACT410": 1, "IMPACT468": 2})
flow["ID 접미사-패널 일치(IM3=341, IM5=410, IM6=468)"] = bool(
    (suffix.map({"IM3": "IMPACT341", "IM5": "IMPACT410", "IM6": "IMPACT468"}) == prim.GENE_PANEL).all())
flow["PID_NUM-패널 순서 Spearman"] = round(float(spearmanr(pid, order).statistic), 3)
flow["PID_NUM 패널별 중앙값"] = pid.groupby(prim.GENE_PANEL).median().astype(int).to_dict()
dec = pd.qcut(pid, 10, labels=[f"D{i}" for i in range(1, 11)])
flow["PID 십분위별 패널 구성"] = pd.crosstab(dec, prim.GENE_PANEL).to_dict(orient="index")
flow["SAMPLE_ORDER>1 비율"] = round(float((tnum > 1).mean()), 4)
lag = prim.AGE_AT_SEQUENCING - prim.AGE_AT_SURGERY
flow["SEQ_LAG_YEARS 중앙값[IQR], 결측"] = [round(float(lag.median()), 2), round(float(lag.quantile(.25)), 2),
                                        round(float(lag.quantile(.75)), 2), int(lag.isna().sum())]
flow["SEQ_LAG_YEARS 음수(0으로 두지 않고 결측 처리)"] = int((lag < 0).sum())
desc = pd.DataFrame({"PID_NUM": pid, "TUMOR_PURITY": prim.TUMOR_PURITY, "SAMPLE_COVERAGE": prim.SAMPLE_COVERAGE,
                     "SEQ_LAG_YEARS": lag, "OS_MONTHS": prim.OS_MONTHS, "GENE_PANEL": prim.GENE_PANEL})
by_panel = desc.groupby("GENE_PANEL").median(numeric_only=True).round(2)
by_panel["N"] = desc.GENE_PANEL.value_counts()
by_panel.to_csv(RESULTS / "artifact_by_panel.tsv", sep="\t")
flow["흔적 간 Spearman"] = desc.drop(columns="GENE_PANEL").corr("spearman").round(2).to_dict()

prim[["SAMPLE_ID", "PATIENT_ID"]].to_csv(RESULTS / "cohort_primary_ids.tsv", sep="\t", index=False)
sens[["SAMPLE_ID", "PATIENT_ID", "SAMPLE_TYPE"]].to_csv(RESULTS / "cohort_sensitivity_ids.tsv", sep="\t", index=False)
(RESULTS / "cohort_flow.json").write_text(json.dumps(flow, ensure_ascii=False, indent=2, default=str))
print(json.dumps(flow, ensure_ascii=False, indent=1, default=str))
print(by_panel)
