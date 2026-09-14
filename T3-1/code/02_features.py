"""02. 특징 묶음 생성: T(원발암), A(수집 흔적), FU(추적; 보조 전용), clinical, genomic, 라벨.

A: PANEL 원-핫, PID_NUM, SAMPLE_ORDER, TUMOR_PURITY, SAMPLE_COVERAGE, SEQ_LAG_YEARS
clinical(B의 일부): 나이, 성별, TMB, FGA, MSI (패널은 A로 옮김)
genomic: 341 공통 유전자 MUT/AMP/DEL/SV, 코호트 빈도 >= 1%
"""
import json

import numpy as np
import pandas as pd

import msk_features as mf
from t31_common import BURDEN_CUT, MIN_FREQ, RESULTS


def artifacts(df):
    A = pd.get_dummies(df["GENE_PANEL"], prefix="PANEL").astype(int)
    A["PID_NUM"] = df["SAMPLE_ID"].str.extract(r"^P-(\d+)-", expand=False).astype(int)
    A["SAMPLE_ORDER"] = df["SAMPLE_ID"].str.extract(r"-T(\d+)-", expand=False).astype(int)
    A["TUMOR_PURITY"] = pd.to_numeric(df["TUMOR_PURITY"], errors="coerce")
    A["SAMPLE_COVERAGE"] = pd.to_numeric(df["SAMPLE_COVERAGE"], errors="coerce")
    lag = pd.to_numeric(df["AGE_AT_SEQUENCING"], errors="coerce") - pd.to_numeric(df["AGE_AT_SURGERY"], errors="coerce")
    A["SEQ_LAG_YEARS"] = lag.where(lag >= 0)
    return A


clin = mf.load_clinical().set_index("SAMPLE_ID", drop=False)
genes = mf.load_common_genes()
info = {}
for name, ids_file in [("primary", "cohort_primary_ids.tsv"), ("sensitivity", "cohort_sensitivity_ids.tsv")]:
    ids = pd.read_csv(RESULTS / ids_file, sep="\t").SAMPLE_ID
    df = clin.loc[ids]
    G = mf.filter_by_frequency(mf.build_genomic_matrix(ids, genes), MIN_FREQ)
    C = mf.clinical_features(df)
    C = C[[c for c in C.columns if not c.startswith("PANEL_")]]
    if name == "sensitivity":
        C["SAMPLE_TYPE_MET"] = (df["SAMPLE_TYPE"] != "Primary").astype(int)
    Y = mf.labels_to_binary(df)
    Y["HIGH_BURDEN"] = (df["MET_SITE_COUNT"] >= BURDEN_CUT).astype(int)
    FU = pd.DataFrame({"LOG_OS_MONTHS": np.log1p(pd.to_numeric(df["OS_MONTHS"], errors="coerce")),
                       "OS_DECEASED": df["OS_STATUS"].astype(str).str.startswith("1").astype(int)})
    bundle = {
        "meta": df[["SAMPLE_ID", "PATIENT_ID", "SAMPLE_TYPE", "CANCER_TYPE", "GENE_PANEL", "MET_SITE_COUNT"]],
        "Y": Y, "T": mf.one_hot(df, "CANCER_TYPE", "CT"), "A": artifacts(df), "FU": FU,
        "clinical": C, "genomic": G,
    }
    pd.to_pickle(bundle, RESULTS / f"features_{name}.pkl")
    info[name] = {"N": len(df), "A": list(bundle["A"].columns), "clinical": list(C.columns),
                  "유전체 특징 수": G.shape[1], "사건 수": Y.sum().to_dict(),
                  "유병률": Y.mean().round(4).to_dict()}
    print(name, json.dumps(info[name], ensure_ascii=False))
(RESULTS / "features_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=int))
