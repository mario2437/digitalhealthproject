"""1단계: 코호트, 부위(노드) 선택, 유전형 군 정의 — 기술 통계만 출력.

네트워크 코호트: NSCLC & IS_DIST_MET_MAPPED & 폐 신경내분비종양(LNET) 제외 (환자당 1검체)
유전형 코호트(주): 위 코호트의 원발 검체
출력: results/cohort.csv, cohort_flow.csv, site_selection.csv, genotype_groups.csv
"""
import re

import numpy as np
import pandas as pd

from common import EXCLUDE_MUT_TYPES, MIN_PREV, RESULTS_DIR, SITE_LABEL, read_tsv

s = read_tsv("data_clinical_sample.txt")
p = read_tsv("data_clinical_patient.txt")
dm_cols = [c for c in s.columns if c.startswith("DMETS_DX_")]

flow = []
d = s[s["CANCER_TYPE"] == "Non-Small Cell Lung Cancer"]
flow.append(("NSCLC 전체 검체", len(d), d["PATIENT_ID"].nunique()))
d = d[d["IS_DIST_MET_MAPPED"] == True]  # noqa: E712
flow.append(("원격전이 부위 매핑됨", len(d), d["PATIENT_ID"].nunique()))
d = d[d["CANCER_TYPE_DETAILED"] != "Lung Neuroendocrine Tumor"]
flow.append(("폐 신경내분비종양 제외 = 네트워크 코호트", len(d), d["PATIENT_ID"].nunique()))
assert d["PATIENT_ID"].is_unique, "환자당 1검체가 아님"
flow.append(("  그중 원발 검체 = 유전형 코호트(주)", int((d["SAMPLE_TYPE"] == "Primary").sum()), None))
flow.append(("  그중 전이 검체", int((d["SAMPLE_TYPE"] == "Metastasis").sum()), None))
d = d.merge(p[["PATIENT_ID", "SEX", "AGE_AT_SEQUENCING"]], on="PATIENT_ID", how="left")

Y = (d[dm_cols] == "Yes").astype(int)
Y.columns = [c.replace("DMETS_DX_", "") for c in dm_cols]
assert (Y.sum(axis=1).values == d["MET_SITE_COUNT"].values).all(), "MET_SITE_COUNT 불일치"

# ---- 부위 선택 (SAP 3절) ----
prev = Y.mean()
prev_primary = Y[d["SAMPLE_TYPE"].values == "Primary"].mean()
sel = pd.DataFrame({"site": prev.index, "label": [SITE_LABEL[k] for k in prev.index],
                    "n": Y.sum().values, "prev": prev.values, "prev_primary": prev_primary.values})
sel["selected"] = (sel["prev"] >= MIN_PREV) & (sel["site"] != "UNSPECIFIED")
sel = sel.sort_values("prev", ascending=False)
sel.to_csv(RESULTS_DIR / "site_selection.csv", index=False)
sites = sel.query("selected")["site"].tolist()

# ---- 유전형 (SAP 4절) ----
ids = set(d["SAMPLE_ID"])
mu = read_tsv("data_mutations.txt")
mu = mu[mu["sampleId"].isin(ids) & ~mu["mutationType"].isin(EXCLUDE_MUT_TYPES)]


def carriers(gene, filt=None):
    m = mu[mu["hugoGeneSymbol"] == gene]
    if filt is not None:
        m = m[m.apply(filt, axis=1)]
    return d["SAMPLE_ID"].isin(set(m["sampleId"])).astype(int).values


def codon(pc):
    mm = re.match(r"^[A-Z*](\d+)", str(pc))
    return int(mm.group(1)) if mm else -1


def egfr_canonical(r):
    pc, t, cd = str(r["proteinChange"]), r["mutationType"], codon(r["proteinChange"])
    if pc in ("L858R", "L861Q", "S768I") or pc.startswith("G719"):
        return True
    return t in ("In_Frame_Del", "In_Frame_Ins") and 729 <= cd <= 761


def kras_canonical(r):
    return r["mutationType"] == "Missense_Mutation" and codon(r["proteinChange"]) in (12, 13, 61)


geno = pd.DataFrame({
    "SAMPLE_ID": d["SAMPLE_ID"].values,
    "EGFR": carriers("EGFR"), "KRAS": carriers("KRAS"),
    "STK11": carriers("STK11"), "KEAP1": carriers("KEAP1"),
    "EGFR_canon": carriers("EGFR", egfr_canonical), "KRAS_canon": carriers("KRAS", kras_canonical),
})
geno["STK11_KEAP1"] = ((geno["STK11"] + geno["KEAP1"]) > 0).astype(int)
geno["grp_main"] = np.select([(geno.EGFR == 1) & (geno.KRAS == 0), (geno.KRAS == 1) & (geno.EGFR == 0)],
                             ["EGFR", "KRAS"], "other")
geno.loc[(geno.EGFR == 1) & (geno.KRAS == 1), "grp_main"] = "both_excluded"
geno["grp_canon"] = np.select([(geno.EGFR_canon == 1) & (geno.KRAS_canon == 0),
                               (geno.KRAS_canon == 1) & (geno.EGFR_canon == 0)], ["EGFR", "KRAS"], "other")
geno["grp_stk"] = np.where(geno.EGFR == 1, "EGFR_excluded", np.where(geno.STK11_KEAP1 == 1, "STK11_KEAP1", "neither"))

hist_map = {"Lung Adenocarcinoma": "adeno", "Lung Squamous Cell Carcinoma": "squamous",
            "Large Cell Neuroendocrine Carcinoma": "lcnec"}
out = pd.DataFrame({
    "SAMPLE_ID": d["SAMPLE_ID"].values, "PATIENT_ID": d["PATIENT_ID"].values,
    "sample_type": d["SAMPLE_TYPE"].values, "met_sample_site": d["METASTATIC_SITE"].values,
    "histology": d["CANCER_TYPE_DETAILED"].map(hist_map).values,
    "panel": d["GENE_PANEL"].values, "sex": d["SEX"].values, "age": d["AGE_AT_SEQUENCING"].values,
    "burden": d["MET_SITE_COUNT"].values.astype(int),
})
assert out["histology"].notna().all()
for k in Y.columns:
    out[f"site_{k}"] = Y[k].values
out["burden_sel"] = Y[sites].sum(axis=1).values
out = out.merge(geno, on="SAMPLE_ID")
out.to_csv(RESULTS_DIR / "cohort.csv", index=False)

fl = pd.DataFrame(flow, columns=["step", "n_samples", "n_patients"])
fl.to_csv(RESULTS_DIR / "cohort_flow.csv", index=False)
print(fl.to_string(index=False))
print(sel.round(3).to_string(index=False))
print("선택 부위", len(sites), sites)
print("부담 분포(MET_SITE_COUNT):", out["burden"].describe().round(2).to_dict())
for st in ["all", "Primary"]:
    o = out if st == "all" else out[out.sample_type == "Primary"]
    print(st, "grp_main", o["grp_main"].value_counts().to_dict(), "| grp_canon", o["grp_canon"].value_counts().to_dict(),
          "| grp_stk", o["grp_stk"].value_counts().to_dict())
print("조직형", out["histology"].value_counts().to_dict(), "연령 결측", int(out["age"].isna().sum()))
