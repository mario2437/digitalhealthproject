"""04. H2: 수집 흔적과 기록 강도(MET_SITE_COUNT)의 연관.

포아송 회귀 + 강건 표준오차(HC0), CANCER_TYPE 고정효과 보정.
  m_panel : MET_SITE_COUNT ~ C(GENE_PANEL, 기준 468) + C(CANCER_TYPE)
  m_fu    : m_panel + LOG_OS_MONTHS + OS_DECEASED  (보조; 추적 기간이 설명하는 부분)
  연속 흔적 단변량(각각 CANCER_TYPE 보정): PID_NUM(1SD), TUMOR_PURITY(10%p), SAMPLE_COVERAGE(100x), SEQ_LAG_YEARS(1년), SAMPLE_ORDER>1
H2 기준: RR(IMPACT341 vs 468) >= 1.10 & CI 하한 > 1.
기술: 상위 5개 원발암 내부 패널별 평균 MET_SITE_COUNT.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from t31_common import RESULTS

b = pd.read_pickle(RESULTS / "features_primary.pkl")
m, A, FU = b["meta"], b["A"], b["FU"]
d = pd.DataFrame({"count": m.MET_SITE_COUNT.values, "panel": m.GENE_PANEL.values, "ctype": m.CANCER_TYPE.values,
                  "pid_sd": ((A.PID_NUM - A.PID_NUM.mean()) / A.PID_NUM.std()).values,
                  "purity10": (A.TUMOR_PURITY / 10).values, "cov100": (A.SAMPLE_COVERAGE / 100).values,
                  "lag": A.SEQ_LAG_YEARS.values, "order2": (A.SAMPLE_ORDER > 1).astype(int).values,
                  "log_os": FU.LOG_OS_MONTHS.values, "dead": FU.OS_DECEASED.values})


def rr_table(fit, terms, model):
    rows = []
    ci = fit.conf_int()
    for t, label in terms.items():
        rows.append(dict(model=model, term=label, RR=np.exp(fit.params[t]), ci_lo=np.exp(ci.loc[t, 0]),
                         ci_hi=np.exp(ci.loc[t, 1]), p=fit.pvalues[t], n=int(fit.nobs)))
    return rows


PANEL_TERMS = {"C(panel, Treatment('IMPACT468'))[T.IMPACT341]": "IMPACT341 vs 468",
               "C(panel, Treatment('IMPACT468'))[T.IMPACT410]": "IMPACT410 vs 468"}
rows = []
f_panel = "count ~ C(panel, Treatment('IMPACT468')) + C(ctype)"
fit = smf.poisson(f_panel, d).fit(cov_type="HC0", disp=0)
rows += rr_table(fit, PANEL_TERMS, "패널+원발암")
fit_crude = smf.poisson("count ~ C(panel, Treatment('IMPACT468'))", d).fit(cov_type="HC0", disp=0)
rows += rr_table(fit_crude, PANEL_TERMS, "패널(보정 없음)")
dfu = d.dropna(subset=["log_os"])
fit_fu = smf.poisson(f_panel + " + log_os + dead", dfu).fit(cov_type="HC0", disp=0)
rows += rr_table(fit_fu, PANEL_TERMS, "패널+원발암+추적(보조)")
rows += rr_table(fit_fu, {"log_os": "log(OS_MONTHS+1) 1단위", "dead": "사망"}, "패널+원발암+추적(보조)")
for v, label in [("pid_sd", "PID_NUM 1SD 증가"), ("purity10", "TUMOR_PURITY 10%p"), ("cov100", "SAMPLE_COVERAGE 100x"),
                 ("lag", "SEQ_LAG_YEARS 1년"), ("order2", "SAMPLE_ORDER>1")]:
    dd = d.dropna(subset=[v])
    f = smf.poisson(f"count ~ {v} + C(ctype)", dd).fit(cov_type="HC0", disp=0)
    rows += rr_table(f, {v: label}, "단변량+원발암")
    f2 = smf.poisson(f"count ~ {v} + C(ctype) + log_os + dead", dd.dropna(subset=["log_os"])).fit(cov_type="HC0", disp=0)
    rows += rr_table(f2, {v: label}, "단변량+원발암+추적(보조)")
rr = pd.DataFrame(rows)
rr.to_csv(RESULTS / "h2_rate_ratios.tsv", sep="\t", index=False, float_format="%.4f")

top = m.CANCER_TYPE.value_counts().head(5).index
tab = d[d.ctype.isin(top)].groupby(["ctype", "panel"])["count"].agg(["mean", "size"]).round(2).unstack("panel")
tab.to_csv(RESULTS / "h2_count_by_panel_top5.tsv", sep="\t")
overall = d.groupby("panel")["count"].agg(["mean", "median", "size"]).round(3)
overall.to_csv(RESULTS / "h2_count_by_panel.tsv", sep="\t")
dec = pd.qcut(A.PID_NUM, 10, labels=False).values + 1
bydec = pd.DataFrame({"decile": dec, "count": d["count"], "os": np.expm1(d.log_os)}).groupby("decile").agg(
    mean_count=("count", "mean"), median_os_months=("os", "median")).round(3)
bydec.to_csv(RESULTS / "h2_count_by_pid_decile.tsv", sep="\t")

main = rr[(rr.model == "패널+원발암") & (rr.term == "IMPACT341 vs 468")].iloc[0]
summary = {"RR(341 vs 468, 원발암 보정)": [round(main.RR, 3), round(main.ci_lo, 3), round(main.ci_hi, 3)],
           "H2 판정 (RR>=1.10 & CI하한>1)": bool(main.RR >= 1.10 and main.ci_lo > 1)}
(RESULTS / "h2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(rr.round(3).to_string())
print(overall, tab, bydec, summary, sep="\n")
