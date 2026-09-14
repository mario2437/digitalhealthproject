"""06. H4: SHAP 설명은 흔적 변수에 얼마나 기대고, 흔적을 통제하면 유전자 설명이 바뀌는가.

전체 주 분석 코호트로 적합 (설명 목적; 성능 평가는 03/05):
  B+A (전체), B (전체, 흔적 제거), B_468 (IMPACT468 단일 패널로 제한 = 층화)
TreeSHAP 평균 |SHAP| → 전체 특징 순위.
H4 기준: B+A에서 흔적 변수 최고 순위 <= 10 인 장기 >= 2/5.
보조: 유전자 특징(MUT/AMP/DEL/SV) 상위 10개 Jaccard — (B+A vs B), (B vs B_468).
탐색(SAP 이후 추가, 이탈로 기록): B_rand = IMPACT468과 같은 크기의 무작위 부분표본으로 적합한 B.
  Jaccard(B vs B_rand)는 '표본 크기만 줄였을 때'의 순위 흔들림 기준선 → B vs B_468과 비교.
"""
import json

import numpy as np
import pandas as pd
import shap

from t31_common import ORGAN_KEYS, OUTCOMES, RESULTS, SEED, design, make_lgbm

b = pd.read_pickle(RESULTS / "features_primary.pkl")
A_cols, G_cols = list(b["A"].columns), list(b["genomic"].columns)
is468 = (b["meta"]["GENE_PANEL"] == "IMPACT468").values
rand_idx = np.sort(np.random.default_rng(SEED).choice(len(is468), int(is468.sum()), replace=False))


def mean_abs_shap(X, y):
    X = X.loc[:, X.nunique() > 1]
    model = make_lgbm().fit(X.values, y)
    sv = shap.TreeExplainer(model).shap_values(X.values)
    if isinstance(sv, list):
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)


def top_genes(imp, k=10):
    return [c for c in imp.index if c in G_cols][:k]


def jaccard(a, b_):
    return len(set(a) & set(b_)) / len(set(a) | set(b_))


rows, summ = [], []
for o in OUTCOMES:
    y = b["Y"][o].values
    imps = {"B+A": mean_abs_shap(design(b, "B+A"), y), "B": mean_abs_shap(design(b, "B"), y),
            "B_468": mean_abs_shap(design(b, "B")[is468], y[is468]),
            "B_rand": mean_abs_shap(design(b, "B").iloc[rand_idx], y[rand_idx])}
    for m, imp in imps.items():
        for r, (f, v) in enumerate(imp.items(), start=1):
            rows.append(dict(outcome=o, 결과=OUTCOMES[o], model=m, feature=f, rank=r, mean_abs_shap=v,
                             group="A" if f in A_cols else "G" if f in G_cols else "T" if f.startswith("CT_") else "C"))
    ba = imps["B+A"]
    art_ranks = {f: int(np.where(ba.index == f)[0][0]) + 1 for f in A_cols if f in ba.index}
    best = min(art_ranks, key=art_ranks.get)
    tg = {m: top_genes(imp) for m, imp in imps.items()}
    summ.append(dict(outcome=o, 결과=OUTCOMES[o], best_artifact=best, best_artifact_rank=art_ranks[best],
                     n_features=len(ba), artifact_ranks=json.dumps(art_ranks),
                     artifact_share_abs_shap=float(ba[[c for c in A_cols if c in ba.index]].sum() / ba.sum()),
                     jaccard_BA_vs_B=jaccard(tg["B+A"], tg["B"]), jaccard_B_vs_B468=jaccard(tg["B"], tg["B_468"]),
                     jaccard_B_vs_Brand=jaccard(tg["B"], tg["B_rand"]),
                     top10_genes_BA=",".join(tg["B+A"]), top10_genes_B=",".join(tg["B"]),
                     top10_genes_B468=",".join(tg["B_468"])))
    print(summ[-1], flush=True)

pd.DataFrame(rows).to_csv(RESULTS / "h4_shap_importance.tsv", sep="\t", index=False, float_format="%.5f")
s = pd.DataFrame(summ)
s.to_csv(RESULTS / "h4_shap_summary.tsv", sep="\t", index=False, float_format="%.4f")
main = s[s.outcome.isin(ORGAN_KEYS)]
summary = {"H4 충족 장기 수 (흔적 최고 순위<=10)": int((main.best_artifact_rank <= 10).sum()),
           "H4 판정 (>=2/5)": bool((main.best_artifact_rank <= 10).sum() >= 2),
           "보조: Jaccard(B+A vs B)<0.7 장기 수": int((main.jaccard_BA_vs_B < 0.7).sum()),
           "보조: Jaccard(B vs B_468)<0.7 장기 수": int((main.jaccard_B_vs_B468 < 0.7).sum())}
(RESULTS / "h4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(s[["결과", "best_artifact", "best_artifact_rank", "artifact_share_abs_shap", "jaccard_BA_vs_B",
         "jaccard_B_vs_B468", "jaccard_B_vs_Brand"]].round(3).to_string())
print(summary)
