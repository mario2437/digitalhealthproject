"""2단계: 주변 동시발생(5.1) vs 전이 부담 보정 동시발생(5.2 주 분석) + 민감도 S1, S3.

출력: results/pairs_main.csv (전체 코호트), pairs_S1_primary.csv, pairs_S3_selburden.csv,
      marginal_vs_adjusted_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from common import (RESULTS_DIR, SITE_LABEL, bh, cmh_test, load_cohort, load_sites, mh_ci, mh_logor,
                    pair_codes, pairs_of, strata_counts)

coh = load_cohort()
sites = load_sites()
pairs = pairs_of(sites)
pidx = [(sites.index(a), sites.index(b)) for a, b in pairs]


def analyse(df, burden_col):
    X = df[[f"site_{s}" for s in sites]].values.astype(int)
    B = df[burden_col].values.astype(int)
    n = len(df)
    rows = []
    for (a, b), (i, j) in zip(pairs, pidx):
        xi, xj = X[:, i], X[:, j]
        t = np.array([[np.sum(xi & xj), np.sum(xi & (1 - xj))], [np.sum((1 - xi) & xj), np.sum((1 - xi) & (1 - xj))]])
        tt = t + 0.5 if (t == 0).any() else t
        lor = np.log(tt[0, 0] * tt[1, 1] / (tt[0, 1] * tt[1, 0]))
        se = np.sqrt((1 / tt).sum())
        p_f = stats.fisher_exact(t)[1]
        pi, pj, pij = xi.mean(), xj.mean(), t[0, 0] / n
        rows.append(dict(site_i=a, site_j=b, label=f"{SITE_LABEL[a]}–{SITE_LABEL[b]}", n=n, n_ij=int(t[0, 0]),
                         prev_i=pi, prev_j=pj, lift=pij / (pi * pj), jaccard=t[0, 0] / (t[0, 0] + t[0, 1] + t[1, 0]),
                         marg_logor=lor, marg_or=np.exp(lor), marg_lo=np.exp(lor - 1.96 * se), marg_hi=np.exp(lor + 1.96 * se),
                         marg_p=p_f))
    out = pd.DataFrame(rows)
    codes = pair_codes(X, B, pidx)
    cnt = strata_counts(codes)
    lor = mh_logor(cnt)
    se, lo, hi = mh_ci(cnt)
    _, p = cmh_test(cnt)
    out["adj_logor"], out["adj_or"], out["adj_lo"], out["adj_hi"], out["adj_p"] = lor, np.exp(lor), np.exp(lo), np.exp(hi), p
    out["marg_fdr"] = bh(out["marg_p"])
    out["adj_fdr"] = bh(out["adj_p"])
    out["marg_sig"] = out["marg_fdr"] < 0.05
    out["adj_sig"] = out["adj_fdr"] < 0.05
    # statsmodels 교차 확인 (첫 쌍)
    k = 0
    tabs = [cnt[k, s, [3, 2, 1, 0]].reshape(2, 2) for s in range(cnt.shape[1]) if cnt[k, s].sum() > 1]
    st = StratifiedTable(tabs)
    assert np.isclose(np.log(st.oddsratio_pooled), lor[k]) and np.isclose(st.test_null_odds(correction=False).pvalue, p[k])
    return out


def summary(out, name):
    d = out["adj_logor"] - out["marg_logor"]
    return dict(analysis=name, n=int(out["n"].iloc[0]), pairs=len(out),
                marg_sig_pos=int((out.marg_sig & (out.marg_logor > 0)).sum()),
                marg_sig_neg=int((out.marg_sig & (out.marg_logor < 0)).sum()),
                adj_sig_pos=int((out.adj_sig & (out.adj_logor > 0)).sum()),
                adj_sig_neg=int((out.adj_sig & (out.adj_logor < 0)).sum()),
                marg_sig_lost=int((out.marg_sig & ~out.adj_sig).sum()),
                sign_flip_sig=int((out.marg_sig & out.adj_sig & (np.sign(out.marg_logor) != np.sign(out.adj_logor))).sum()),
                median_diff_logor=float(np.median(d)), median_marg_logor=float(out.marg_logor.median()),
                median_adj_logor=float(out.adj_logor.median()),
                n_marg_pos_nominal=int((out.marg_logor > 0).sum()), n_adj_pos=int((out.adj_logor > 0).sum()))


res = {
    "main": (coh, "burden"),
    "S1_primary": (coh[coh.sample_type == "Primary"], "burden"),
    "S3_selburden": (coh, "burden_sel"),
}
summ = []
for name, (df, bcol) in res.items():
    out = analyse(df.reset_index(drop=True), bcol)
    out.to_csv(RESULTS_DIR / f"pairs_{name}.csv", index=False)
    summ.append(summary(out, name))
    if name == "main":
        print(out.sort_values("adj_p")[["label", "n_ij", "lift", "marg_or", "marg_fdr", "adj_or", "adj_lo", "adj_hi", "adj_fdr"]]
              .round(3).to_string(index=False))
sm = pd.DataFrame(summ)
sm.to_csv(RESULTS_DIR / "marginal_vs_adjusted_summary.csv", index=False)
print(sm.T.to_string())
