"""07. 완화 전략 비교 (기술 분석).

같은 두 질문을 네 가지 평가 방식으로 추정:
  Δ_A = AUROC(B+A) − AUROC(B)   : 흔적 변수의 겉보기 이득
  Δ_G = AUROC(B)   − AUROC(T)   : 유전체·임상의 겉보기 이득
  M0 무작위 5-겹 CV (03 OOF)
  M1 패널 층화 AUROC (M0 OOF를 패널 층 내부에서 평가)
  M2 IMPACT468 단일 패널 제한 5-겹 CV (PANEL 제외)
  M3 시간 분할 시험 세트 (05 예측) — 기준값
판정 규칙(SAP): M0~M2 중 Δ_A가 M3에 가장 가까운 방식 = 가장 정직한 내부 평가.
               M0의 Δ_A가 M1 또는 M3에서 50% 이상 줄면 '흔적에 의한 부풀림'.
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from t31_common import (N_JOBS, ORGAN_KEYS, OUTCOMES, RESULTS, delta_auc_ci, design, fast_auc, oof_predict,
                        stratified_auc)

b = pd.read_pickle(RESULTS / "features_primary.pkl")
oof = pd.read_csv(RESULTS / "oof_primary.tsv", sep="\t")
panel = b["meta"]["GENE_PANEL"].values
is468 = panel == "IMPACT468"
te = pd.read_csv(RESULTS / "h3_test_predictions.tsv", sep="\t")
te = te[te.split == "temporal"].set_index("SAMPLE_ID")
te.index = te.index.astype(str)
Yall = b["Y"].set_axis(pd.Index(b["meta"]["SAMPLE_ID"].astype(str).tolist()), axis=0)

oof468 = pd.DataFrame(index=np.where(is468)[0])
for o in OUTCOMES:
    y = b["Y"][o].values[is468]
    for sp in ["T", "B", "B+A"]:
        X = design(b, sp, drop_panel=True)[is468]
        oof468[f"{o}|{sp}"] = oof_predict(X.loc[:, X.nunique() > 1], y)
    print("M2", o, flush=True)

jobs = []  # (strategy, outcome, contrast, y, s_new, s_ref, strata)
for o in OUTCOMES:
    y = b["Y"][o].values
    yte = Yall.loc[te.index, o].values
    y468 = y[is468]
    for name, new, ref in [("Δ_A", "B+A", "B"), ("Δ_G", "B", "T")]:
        jobs.append(("M0 무작위 CV", o, name, new, y, oof[f"{o}|{new}"].values, oof[f"{o}|{ref}"].values, None))
        jobs.append(("M1 패널 층화", o, name, new, y, oof[f"{o}|{new}"].values, oof[f"{o}|{ref}"].values, panel))
        jobs.append(("M2 IMPACT468 제한", o, name, new, y468, oof468[f"{o}|{new}"].values, oof468[f"{o}|{ref}"].values, None))
        jobs.append(("M3 시간 분할 시험", o, name, new, yte, te[f"{o}|{new}"].values, te[f"{o}|{ref}"].values, None))
res = Parallel(n_jobs=N_JOBS)(delayed(delta_auc_ci)(y, sn, sr, st) for *_, y, sn, sr, st in jobs)
rows = []
for (strat, o, name, new, y, sn, sr, st), (p, lo, hi) in zip(jobs, res):
    auc_new = stratified_auc(y, sn, st) if st is not None else fast_auc(y, sn)
    rows.append(dict(strategy=strat, outcome=o, 결과=OUTCOMES[o], contrast=name, delta=p, ci_lo=lo, ci_hi=hi,
                     auroc_new_model=auc_new, new_model=new, n=len(y)))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "m_mitigation.tsv", sep="\t", index=False, float_format="%.4f")

wide = df[df.contrast == "Δ_A"].pivot(index="outcome", columns="strategy", values="delta")
ref = wide["M3 시간 분할 시험"]
dist = {s: float((wide[s] - ref).loc[ORGAN_KEYS].abs().mean()) for s in wide.columns if not s.startswith("M3")}
m0 = wide["M0 무작위 CV"].loc[ORGAN_KEYS]
shrink = lambda s: int(((m0 > 0) & (wide[s].loc[ORGAN_KEYS] <= 0.5 * m0)).sum())
summary = {"Δ_A의 M3와 평균 절대 차이(5장기)": {k: round(v, 4) for k, v in dist.items()},
           "가장 정직한 내부 평가(M3에 가장 가까움)": min(dist, key=dist.get),
           "M0 대비 Δ_A 50% 이상 감소 장기 수: M1": shrink("M1 패널 층화"),
           "M0 대비 Δ_A 50% 이상 감소 장기 수: M3": shrink("M3 시간 분할 시험")}
(RESULTS / "m_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(df.pivot_table(index=["contrast", "결과"], columns="strategy", values="delta").round(4).to_string())
print(summary)
