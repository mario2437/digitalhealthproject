"""03. H1: 수집 흔적 변수만으로 전이 장기/부담을 예측할 수 있는가.

주 분석 코호트, 6개 결과(5장기 + HIGH_BURDEN) × 모델 A, T, T+A, B, B+A (5-겹 OOF).
- AUROC, AUPRC, 패널 층화 AUROC
- ΔAUROC(T+A − T): 일반 / 패널 층화, 쌍체 부트스트랩 95% CI
- H1 기준: ΔAUROC(T+A−T) >= 0.02 & CI 하한 > 0 인 장기 >= 2/5 ; 보조: A 단독 AUROC >= 0.60 인 장기 >= 2/5
민감도: Primary+Metastasis 코호트에서 T, T+A (SAMPLE_TYPE 지표를 T에 추가).
"""
import json
import time

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

from t31_common import (N_JOBS, ORGAN_KEYS, OUTCOMES, RESULTS, delta_auc_ci, design, fast_auc, oof_predict,
                        stratified_auc)

SPECS = ["A", "T", "T+A", "B", "B+A"]


def run_oof(bundle, specs, extra_T=None):
    out = pd.DataFrame({"SAMPLE_ID": bundle["meta"]["SAMPLE_ID"].values})
    for o in OUTCOMES:
        y = bundle["Y"][o].values
        for sp in specs:
            t = time.time()
            X = design(bundle, sp)
            if extra_T is not None and "T" in sp:
                X = pd.concat([X, extra_T.astype(float)], axis=1)
            out[f"{o}|{sp}"] = oof_predict(X, y)
            print(f"{o} {sp} X={X.shape} {time.time() - t:.1f}s", flush=True)
    return out


prim = pd.read_pickle(RESULTS / "features_primary.pkl")
oof = run_oof(prim, SPECS)
oof.to_csv(RESULTS / "oof_primary.tsv", sep="\t", index=False)
panel = prim["meta"]["GENE_PANEL"].values

perf = []
for o in OUTCOMES:
    y = prim["Y"][o].values
    for sp in SPECS:
        s = oof[f"{o}|{sp}"].values
        perf.append(dict(outcome=o, 결과=OUTCOMES[o], model=sp, auroc=fast_auc(y, s),
                         auroc_panel_strat=stratified_auc(y, s, panel), auprc=average_precision_score(y, s),
                         prevalence=y.mean()))
perf = pd.DataFrame(perf)
perf.to_csv(RESULTS / "h1_perf_primary.tsv", sep="\t", index=False, float_format="%.4f")

jobs = []
for o in OUTCOMES:
    y = prim["Y"][o].values
    for new, ref in [("T+A", "T"), ("B+A", "B"), ("B", "T")]:
        for strat in [False, True]:
            jobs.append((o, new, ref, strat, y))
res = Parallel(n_jobs=N_JOBS)(delayed(delta_auc_ci)(y, oof[f"{o}|{n}"].values, oof[f"{o}|{r}"].values,
                                                   panel if st else None) for o, n, r, st, y in jobs)
delta = pd.DataFrame([dict(outcome=o, 결과=OUTCOMES[o], comparison=f"{n} - {r}", panel_stratified=st,
                           delta=p, ci_lo=lo, ci_hi=hi) for (o, n, r, st, _), (p, lo, hi) in zip(jobs, res)])
delta.to_csv(RESULTS / "h1_delta_primary.tsv", sep="\t", index=False, float_format="%.4f")

# 민감도 코호트
sens = pd.read_pickle(RESULTS / "features_sensitivity.pkl")
st = sens["clinical"][["SAMPLE_TYPE_MET"]]
oof_s = run_oof(sens, ["T", "T+A"], extra_T=st)
oof_s.to_csv(RESULTS / "oof_sensitivity.tsv", sep="\t", index=False)
res_s = Parallel(n_jobs=N_JOBS)(delayed(delta_auc_ci)(sens["Y"][o].values, oof_s[f"{o}|T+A"].values,
                                                     oof_s[f"{o}|T"].values) for o in OUTCOMES)
delta_s = pd.DataFrame([dict(outcome=o, 결과=OUTCOMES[o], auroc_T=fast_auc(sens["Y"][o].values, oof_s[f"{o}|T"].values),
                             auroc_TA=fast_auc(sens["Y"][o].values, oof_s[f"{o}|T+A"].values),
                             delta=p, ci_lo=lo, ci_hi=hi) for o, (p, lo, hi) in zip(OUTCOMES, res_s)])
delta_s.to_csv(RESULTS / "h1_delta_sensitivity.tsv", sep="\t", index=False, float_format="%.4f")

d_main = delta[(delta.comparison == "T+A - T") & (~delta.panel_stratified) & delta.outcome.isin(ORGAN_KEYS)]
a_only = perf[(perf.model == "A") & perf.outcome.isin(ORGAN_KEYS)]
summary = {
    "H1 주 기준 충족 장기 수 (Δ>=0.02 & CI하한>0)": int(((d_main.delta >= 0.02) & (d_main.ci_lo > 0)).sum()),
    "H1 주 기준 판정 (>=2/5)": bool(((d_main.delta >= 0.02) & (d_main.ci_lo > 0)).sum() >= 2),
    "H1 보조: A 단독 AUROC>=0.60 장기 수": int((a_only.auroc >= 0.60).sum()),
    "H1 보조 판정 (>=2/5)": bool((a_only.auroc >= 0.60).sum() >= 2),
}
(RESULTS / "h1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(perf.pivot(index="결과", columns="model", values="auroc").round(3))
print(perf.pivot(index="결과", columns="model", values="auroc_panel_strat").round(3))
print(delta.round(4).to_string())
print(delta_s.round(4).to_string())
print(summary)
