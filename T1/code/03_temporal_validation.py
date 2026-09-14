"""03. 시간적 검증: 2016-2018 기준 모형을 2019-2024 각 연도에 고정 적용.

결과변수: composite(주), fpg(FPG 단독), hba1c(HbA1c 단독)
시나리오: S0 원자료, S1a HbA1c 전환식, S1b 혈압 전환식, S1 둘 다 (구성요소 결과는 S0·S1만)
지표: AUROC, AUPRC, mbc, O/E, 보정 절편·기울기, ICI (+부트스트랩 95% CI), ICI 영점(완전 보정 모의)
기준 성능: 2016-2018 5겹 CV 예측값
산출: results/yearly_metrics.csv, reference_cv.csv, ref_model_coef.csv
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from common import (EVAL_YEARS, N_JOBS, PRED, REF_YEARS, SEED, VAR_KO, boot_indices, citl, ici,
                    ici_null, out_dirs, pct_ci, point_metrics, scenario)
from models import cv_pred_lr, fit_lr, pred_lr

RES, _ = out_dirs()
SCEN = {"S0": dict(hba1c_conv=False, bp_conv=False), "S1a": dict(hba1c_conv=True, bp_conv=False),
        "S1b": dict(hba1c_conv=False, bp_conv=True), "S1": dict(hba1c_conv=True, bp_conv=True)}


def boot_one(y, p, idx):
    yb, pb = y[idx], p[idx]
    return (roc_auc_score(yb, pb), citl(yb, pb), ici(yb, pb), yb.mean() / pb.mean())


def job(outcome, scen, year, y, p):
    pm = point_metrics(y, p)
    nmed, n95 = ici_null(p, seed=SEED + year)
    bs = np.array([boot_one(y, p, i) for i in boot_indices(len(y), seed=SEED + year)])
    row = dict(outcome=outcome, scenario=scen, year=year, **pm, ICI_null=nmed, ICI_null95=n95,
               excess_ICI=pm["ICI"] - nmed)
    for j, k in enumerate(["AUROC", "CITL", "ICI", "OE"]):
        row[f"{k}_lo"], row[f"{k}_hi"] = pct_ci(bs[:, j])
    return row


def main():
    base = pd.read_pickle(RES / "base_cohort.pkl")
    base = base[~base["undx_harm"] & base["cc"]].reset_index(drop=True)
    tasks, ref_rows = [], []
    for outcome in ["composite", "fpg", "hba1c"]:
        scens = SCEN if outcome == "composite" else {k: SCEN[k] for k in ("S0", "S1")}
        ref = scenario(base[base["year"].isin(REF_YEARS)], outcome=outcome)
        m = fit_lr(ref)
        if outcome == "composite":
            ci = m.conf_int()
            pd.DataFrame({"var": m.params.index, "label": [VAR_KO.get(v, v) for v in m.params.index],
                          "beta": m.params.values, "OR": np.exp(m.params.values),
                          "OR_lo": np.exp(ci[0].values), "OR_hi": np.exp(ci[1].values),
                          "p": m.pvalues.values}).to_csv(RES / "ref_model_coef.csv", index=False)
        pcv = cv_pred_lr(ref)
        nmed, n95 = ici_null(pcv, seed=SEED)
        pm = point_metrics(ref["y"].values, pcv)
        ref_rows.append(dict(outcome=outcome, **pm, ICI_null=nmed, ICI_null95=n95,
                             excess_ICI=pm["ICI"] - nmed))
        for sname, kw in scens.items():
            d = scenario(base, outcome=outcome, **kw)
            for yr in EVAL_YEARS:
                t = d[d["year"] == yr]
                tasks.append((outcome, sname, yr, t["y"].values, pred_lr(m, t, PRED)))
    rows = Parallel(n_jobs=N_JOBS)(delayed(job)(*a) for a in tasks)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "yearly_metrics.csv", index=False)
    refdf = pd.DataFrame(ref_rows)
    refdf.to_csv(RES / "reference_cv.csv", index=False)
    pd.set_option("display.width", 250)
    print(refdf.round(3).to_string())
    cols = ["outcome", "scenario", "year", "n", "prev", "mean_pred", "OE", "AUROC", "mbc", "CITL",
            "slope", "ICI", "ICI_null", "excess_ICI"]
    print(out[cols].round(3).to_string())


if __name__ == "__main__":
    main()
