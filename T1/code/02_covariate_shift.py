"""02. 공변량·예측변수 측정 이동 진단(결과변수 사용 안 함).

- 도메인 분류기 AUROC(기준 2016-2018 vs 각 평가 연도; 로지스틱, 민감도 LightGBM), 원자료/혈압 전환식
- 예측변수별 SMD(기준 대비)
- 잔차 진단: 허리둘레 ~ BMI·연령·성별(2019 줄자 변경), 수축기혈압 ~ 연령·성별·BMI(2021 혈압계 변경)
- 수면시간 연도별 평균(문항 형식 순환)
산출: results/domain_auc.csv, smd.csv, residual_diag.csv, predictor_means.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import EVAL_YEARS, PRED, REF_YEARS, out_dirs, scenario
from models import domain_auc, smd

RES, _ = out_dirs()


def main():
    base = pd.read_pickle(RES / "base_cohort.pkl")
    base = base[~base["undx_harm"] & base["cc"]]
    rows, smd_rows = [], []
    for bp_conv in (False, True):
        d = scenario(base, bp_conv=bp_conv)
        ref = d[d["year"].isin(REF_YEARS)]
        for y in EVAL_YEARS:
            t = d[d["year"] == y]
            rows.append(dict(year=y, bp_conv=bp_conv, n_ref=len(ref), n_year=len(t),
                             domain_auc_lr=domain_auc(ref, t, kind="lr"),
                             domain_auc_lgb=domain_auc(ref, t, kind="lgb")))
            for v in PRED:
                smd_rows.append(dict(year=y, bp_conv=bp_conv, var=v, smd=smd(ref[v], t[v])))
    dom = pd.DataFrame(rows)
    dom.to_csv(RES / "domain_auc.csv", index=False)
    pd.DataFrame(smd_rows).to_csv(RES / "smd.csv", index=False)
    print(dom.round(3).to_string())

    # 잔차 진단
    d = base.copy()
    d["sbp_h"], _ = scenario(base, bp_conv=True)[["sbp", "dbp"]].T.values
    ref = d[d["year"].isin(REF_YEARS)]
    specs = {"wc": "wc ~ bmi * male + age", "HE_sbp": "HE_sbp ~ age * male + bmi",
             "sbp_h": "sbp_h ~ age * male + bmi"}
    res_rows = []
    for var, f in specs.items():
        fr = f.replace("sbp_h ~", "HE_sbp ~") if var == "sbp_h" else f
        m = smf.ols(fr, data=ref).fit()
        for y, g in d.groupby("year"):
            r = g[var] - m.predict(g)
            res_rows.append(dict(var=var, year=y, n=len(g), mean_resid=r.mean(),
                                 lo=r.mean() - 1.96 * r.std(ddof=1) / np.sqrt(len(r)),
                                 hi=r.mean() + 1.96 * r.std(ddof=1) / np.sqrt(len(r))))
    rd = pd.DataFrame(res_rows)
    rd.to_csv(RES / "residual_diag.csv", index=False)
    print(rd.round(2).to_string())

    means = base.groupby("year").agg(**{v: (v, "mean") for v in PRED},
                                     sleep_item=("sleep_item", "first"))
    means.to_csv(RES / "predictor_means.csv")
    print(means.round(2).to_string())


if __name__ == "__main__":
    main()
