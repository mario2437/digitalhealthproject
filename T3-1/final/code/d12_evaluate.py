"""D1·D2 판정: d1_metrics.tsv → 사전 규칙(R1–R3, HD1–HD6) 판정, 주 컷 ΔAUROC(Sk − S1) CI.
출력: results/d1_strategy_table_c70.tsv, d1_delta_vs_S1_c70.tsv, d2_internal_validity.tsv, d12_summary.json
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from deep_common import MAX_JOBS, ORGAN_KEYS, OUTCOMES, RESULTS, delta_auc_ci, load_bundle

m = pd.read_csv(RESULTS / "d1_metrics.tsv", sep="\t")
b = load_bundle()
S_NEW = ["S1b", "S2", "S2b", "S3", "S4", "S5a", "S5b", "S6", "S7"]
summ = {}

# ---------- D1: 주 컷 전략 표
c70 = m[m.cut == "c70"].copy()
c70["abs_gap"] = c70.gap_E0.abs()
c70["abs_lnoe"] = np.log(c70.oe_test).abs()
org = c70[c70.outcome.isin(ORGAN_KEYS)]
tab = org.groupby("strategy").agg(auc_E0=("auc_E0", "mean"), auc_test=("auc_test", "mean"),
                                  mean_abs_gap=("abs_gap", "mean"), mean_abs_lnoe=("abs_lnoe", "mean"),
                                  oe_test=("oe_test", "mean"), cal_slope=("cal_slope", "mean"),
                                  auc_test_complete=("auc_test_complete", "mean"),
                                  oe_test_complete=("oe_test_complete", "mean"), auc_E3=("auc_E3", "mean"))
s1 = org[org.strategy == "S1"].set_index("outcome")
rules = {}
for s in ["S0"] + S_NEW:
    x = org[org.strategy == s].set_index("outcome")
    r1 = int(((x.auc_test - s1.auc_test) >= -0.005).sum())
    rules[s] = dict(R1_n=r1, R1=r1 >= 4, R2=bool(x.abs_gap.mean() < s1.abs_gap.mean()),
                    R3=bool(x.abs_lnoe.mean() < s1.abs_lnoe.mean()))
tab = tab.join(pd.DataFrame(rules).T)
tab.to_csv(RESULTS / "d1_strategy_table_c70.tsv", sep="\t", float_format="%.4f")
cand = [s for s in S_NEW if rules[s]["R1"] and (rules[s]["R2"] or rules[s]["R3"])]
if cand:
    best = sorted(cand, key=lambda s: (-(rules[s]["R2"] + rules[s]["R3"]), tab.loc[s, "mean_abs_gap"]))[0]
else:
    best = "S1 (추가 기법 이득 없음)"
summ["D1 규칙"] = rules
summ["D1 권고 전략"] = best

# ΔAUROC(Sk − S1) CI (주 컷, 시험 세트)
pred = pd.read_csv(RESULTS / "d1_test_pred_c70.tsv.gz", sep="\t")
Y = b["Y"].set_axis(b["meta"]["SAMPLE_ID"].values).loc[pred.SAMPLE_ID]
jobs = [(o, s) for o in OUTCOMES for s in ["S0"] + S_NEW]
res = Parallel(n_jobs=MAX_JOBS)(delayed(delta_auc_ci)(Y[o].values, pred[f"{o}|{s}"].values,
                                                     pred[f"{o}|S1"].values) for o, s in jobs)
dl = pd.DataFrame([dict(outcome=o, 결과=OUTCOMES[o], strategy=s, delta=p, ci_lo=lo, ci_hi=hi)
                   for (o, s), (p, lo, hi) in zip(jobs, res)])
dl.to_csv(RESULTS / "d1_delta_vs_S1_c70.tsv", sep="\t", index=False, float_format="%.4f")

# 가설
hd1 = {}
for c in ["c50", "c60", "c80"]:
    x = m[(m.cut == c) & m.outcome.isin(ORGAN_KEYS)]
    g = x.assign(ag=x.gap_E0.abs()).groupby("strategy").ag.mean()
    hd1[c] = dict(S0=round(g["S0"], 4), S1=round(g["S1"], 4), S0_gt_S1=bool(g["S0"] > g["S1"]))
summ["HD1"] = dict(detail=hd1, 판정=sum(v["S0_gt_S1"] for v in hd1.values()) >= 2)
d5b = dl[(dl.strategy == "S5b") & dl.outcome.isin(ORGAN_KEYS)]
summ["HD2"] = dict(CI하한양수_장기=int((d5b.ci_lo > 0).sum()), 판정=bool((d5b.ci_lo > 0).sum() >= 2))
s2 = org[org.strategy == "S2"].set_index("outcome")
hd3a = int(((s2.auc_test - s1.auc_test) >= 0.005).sum())
hd3b = int((np.log(s1.oe_test_complete).abs() < np.log(s1.oe_test).abs()).sum())
summ["HD3"] = dict(a_n=hd3a, a판정=hd3a >= 3, b_n=hd3b, b판정=hd3b >= 3)
summ["HD4"] = dict(S4_R1=rules["S4"]["R1"], S4_R2=rules["S4"]["R2"], 판정=rules["S4"]["R1"] and rules["S4"]["R2"])

# ---------- D2: 내부 평가 방식 타당성
rows = []
for c in ["c50", "c60", "c70", "c80"]:
    for o in ORGAN_KEYS:
        x = m[(m.cut == c) & (m.outcome == o)].set_index("strategy")
        ext_d = x.loc["S0", "auc_test"] - x.loc["S1", "auc_test"]
        for e in ["E0", "E1", "E2", "E3"]:
            int_d = x.loc["S0", f"auc_{e}"] - x.loc["S1", f"auc_{e}"]
            rows.append(dict(cut=c, outcome=o, method=e, target="delta_A", internal=int_d, external=ext_d))
            for s in ["S0", "S1"]:
                rows.append(dict(cut=c, outcome=o, method=e, target=f"auc_{s}", internal=x.loc[s, f"auc_{e}"],
                                 external=x.loc[s, "auc_test"]))
d2 = pd.DataFrame(rows)
d2["err"] = d2.internal - d2.external
d2.to_csv(RESULTS / "d2_internal_validity.tsv", sep="\t", index=False, float_format="%.4f")
da = d2[d2.target == "delta_A"]
mae_d = da.groupby("method").err.apply(lambda e: e.abs().mean())
me_d = da.groupby("method").err.mean()
agree = da.groupby("method").apply(lambda g: ((g.internal >= 0.01) == (g.external >= 0.01)).mean(),
                                   include_groups=False)
lv = d2[d2.target != "delta_A"]
mae_l = lv.groupby("method").err.apply(lambda e: e.abs().mean())
me_l = lv.groupby("method").err.mean()
summ["D2"] = {"Δ_A MAE": mae_d.round(4).to_dict(), "Δ_A 평균오차(+과대)": me_d.round(4).to_dict(),
              "흔적추가 권고 일치율(≥0.01)": agree.round(3).to_dict(),
              "AUROC 수준 MAE(S0·S1)": mae_l.round(4).to_dict(), "AUROC 수준 평균오차": me_l.round(4).to_dict(),
              "외부 Δ_A 범위": [round(da.external.min(), 4), round(da.external.max(), 4)]}
summ["HD5"] = dict(판정=bool(mae_d["E3"] <= 0.5 * mae_d["E0"]))
summ["HD6"] = dict(판정=bool(mae_l["E3"] < mae_l["E0"]))
# 보조: 전략별 E0/E3의 외부 AUROC 오차(4컷×5장기)
mo = m[m.outcome.isin(ORGAN_KEYS)]
summ["보조: 전략별 |E0−시험| MAE"] = mo.assign(e=(mo.auc_E0 - mo.auc_test).abs()).groupby("strategy").e.mean().round(4).to_dict()
summ["보조: 전략별 |E3−시험| MAE"] = mo.assign(e=(mo.auc_E3 - mo.auc_test).abs()).groupby("strategy").e.mean().round(4).to_dict()
(RESULTS / "d12_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str))
print(tab.round(4).to_string())
print(dl.pivot(index="strategy", columns="결과", values="delta").round(4).to_string())
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
