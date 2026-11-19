"""D4 (탐색, 사전 선언): 패널/등록 순번과 기록 장기 수의 연관이 추적 기간과 대상자 선택 중 무엇과 더 연관되는가.

포아송 회귀(HC0), MET_SITE_COUNT ~ 패널(기준 468) + CANCER_TYPE + ...
 (a) 원발암만  (b) + cr(OS,df=4)*사망  (c) 사망자만 + cr(OS,df=4)  (d) OS>=36개월 + cr(OS,df=4)*사망
 (e) (b) + 시퀀싱 시점 전이 확인(MET_AT_SEQ: 예/아니오/결측)
PID_NUM(1SD) 추세도 같은 모형으로. 기술: PID 십분위별 사망 비율·OS 중앙값·MET_AT_SEQ 비율.
출력: results/d4_rr.tsv, d4_by_decile.tsv, d4_summary.json
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from deep_common import RESULTS, load_bundle, V2_CODE  # noqa: F401
import msk_features as mf

b = load_bundle()
clin = mf.load_clinical().set_index("SAMPLE_ID").loc[b["meta"]["SAMPLE_ID"]]
d = pd.DataFrame({
    "count": b["meta"]["MET_SITE_COUNT"].values,
    "ct": b["meta"]["CANCER_TYPE"].values,
    "panel": pd.Categorical(b["meta"]["GENE_PANEL"].values, categories=["IMPACT468", "IMPACT341", "IMPACT410"]),
    "os": np.expm1(b["FU"]["LOG_OS_MONTHS"].values),
    "dead": b["FU"]["OS_DECEASED"].values,
    "pid": b["A"]["PID_NUM"].values,
})
d["pid_z"] = (d.pid - d.pid.mean()) / d.pid.std()
ev = pd.to_numeric(clin["AGE_AT_EVIDENCE_OF_METS"], errors="coerce").values
sq = pd.to_numeric(clin["AGE_AT_SEQUENCING"], errors="coerce").values
d["met_at_seq"] = np.where(np.isnan(ev) | np.isnan(sq), "NA", np.where(ev <= sq, "yes", "no"))
d["decile"] = pd.qcut(d.pid, 10, labels=[f"D{i}" for i in range(1, 11)])
d = d.dropna(subset=["os"]).reset_index(drop=True)

FU = "cr(os, df=4)*dead"
MODELS = {
    "a 원발암": (d, "C(ct)"),
    "b +추적 스플라인": (d, f"C(ct) + {FU}"),
    "c 사망자만 +OS 스플라인": (d[d.dead == 1], "C(ct) + cr(os, df=4)"),
    "d OS>=36개월 +추적 스플라인": (d[d.os >= 36], f"C(ct) + {FU}"),
    "e +추적 +시퀀싱 시점 전이": (d, f"C(ct) + {FU} + C(met_at_seq)"),
}
rows = []
for name, (dd, cov) in MODELS.items():
    dd = dd[dd.ct.map(dd.ct.value_counts()) >= 20]
    for expo, terms in [("panel", ["panel[T.IMPACT341]", "panel[T.IMPACT410]"]), ("pid_z", ["pid_z"])]:
        f = sm.families.Poisson()
        fit = smf.glm(f"count ~ {expo} + {cov}", data=dd, family=f).fit(cov_type="HC0")
        for t in terms:
            lo, hi = fit.conf_int().loc[t]
            rows.append(dict(model=name, n=len(dd), term=t, rr=np.exp(fit.params[t]), ci_lo=np.exp(lo),
                             ci_hi=np.exp(hi)))
    print(name, len(dd), flush=True)
rr = pd.DataFrame(rows)
rr.to_csv(RESULTS / "d4_rr.tsv", sep="\t", index=False, float_format="%.4f")

dec = d.groupby("decile", observed=True).agg(n=("count", "size"), mean_count=("count", "mean"),
                                            dead=("dead", "mean"), os_median=("os", "median"),
                                            met_at_seq_yes=("met_at_seq", lambda s: (s == "yes").mean()),
                                            met_at_seq_na=("met_at_seq", lambda s: (s == "NA").mean()))
dec.to_csv(RESULTS / "d4_by_decile.tsv", sep="\t", float_format="%.4f")

p341 = rr[rr.term == "panel[T.IMPACT341]"].set_index("model")
c, dd_ = p341.loc["c 사망자만 +OS 스플라인"], p341.loc["d OS>=36개월 +추적 스플라인"]
if all(x.rr >= 1.10 and x.ci_lo > 1 for x in (c, dd_)):
    verdict = "추적 기간 차이만으로 설명되지 않음(대상자 선택 등 시기 차이도 연관)"
elif all(x.rr < 1.05 for x in (c, dd_)):
    verdict = "추적 기간 차이가 주로 연관"
else:
    verdict = "구분 불가"
lb, le, la = (np.log(p341.loc[k, "rr"]) for k in ["b +추적 스플라인", "e +추적 +시퀀싱 시점 전이", "a 원발암"])
summary = {"RR341 by model": p341.rr.round(3).to_dict(), "해석 규칙 판정": verdict,
           "a→b 로그RR 감소 비율(추적)": round(float(1 - lb / la), 3),
           "b→e 로그RR 추가 감소 비율(선택 대리)": round(float((lb - le) / la), 3)}
(RESULTS / "d4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(rr.round(3).to_string())
print(dec.round(3).to_string())
print(summary)
