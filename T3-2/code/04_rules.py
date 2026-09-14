"""4단계: 연관 규칙 (보조, SAP 5.3) + 부위별 부담 기울기 진단(탐색, 사후 추가).

선행 1–2개 부위 → 결과 1개 부위, 최소 지지도 0.05, 최소 신뢰도 0.5.
출력: results/rules_all.csv, rules_top.csv, site_prev_by_burden.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SITE_LABEL, load_cohort, load_sites

MIN_SUP, MIN_CONF = 0.05, 0.5
coh = load_cohort()
sites = load_sites()
X = coh[[f"site_{s}" for s in sites]].astype(bool)
X.columns = sites
n = len(X)
rows = []
for k in (1, 2):
    for ante in combinations(sites, k):
        A = X[list(ante)].all(axis=1)
        if A.mean() < MIN_SUP:
            continue
        for cons in sites:
            if cons in ante:
                continue
            both = A & X[cons]
            sup, conf = both.mean(), both.sum() / A.sum()
            if sup >= MIN_SUP and conf >= MIN_CONF:
                # 규칙 적용 환자와 비적용 환자의 전이 부담 (해석 참고용)
                rows.append(dict(antecedent=" + ".join(SITE_LABEL[a] for a in ante), consequent=SITE_LABEL[cons],
                                 cons_code=cons, k=k, support=sup, confidence=conf, lift=conf / X[cons].mean(),
                                 n_antecedent=int(A.sum()), median_burden_antecedent=float(coh.loc[A.values, "burden"].median())))
rules = pd.DataFrame(rows).sort_values("lift", ascending=False)
rules.to_csv(RESULTS_DIR / "rules_all.csv", index=False)
top = pd.concat([rules.head(5).assign(set="top5_lift"),
                 rules[rules.cons_code == "CNS_BRAIN"].head(3).assign(set="top3_brain")])
top.to_csv(RESULTS_DIR / "rules_top.csv", index=False)
print("규칙 수", len(rules), "| 전체 부담 중앙값", coh["burden"].median())
print(top.round(3).to_string(index=False))

# 탐색(사후): 전이 부담 층별 부위 유병률 — curveball 과 MH 결과 차이 해석용
coh["bcat"] = np.minimum(coh["burden"], 7)
tab = coh.groupby("bcat")[[f"site_{s}" for s in sites]].mean()
tab.columns = sites
tab.insert(0, "n", coh.groupby("bcat").size())
tab.to_csv(RESULTS_DIR / "site_prev_by_burden.csv")
print(tab.round(2).to_string())
