"""5단계: 유전형별 부담 보정 네트워크 비교 (SAP 5.4) + 민감도 S4, S5 + 보조(STK11/KEAP1).

간선 가중치 w = 나머지 부담 층화 MH log OR. Δ = w_A − w_B.
순열: 층(조직형×패널[×검체 유형]) 안에서 군 라벨 섞기, B = 2000, seed 2026.
출력: results/geno_edges_<비교>.csv, geno_global.csv, geno_burden_poisson.csv, geno_site_prev.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (GENO_MIN_PREV, N_PERM, RESULTS_DIR, SEED, SITE_LABEL, bh, load_cohort, load_sites,
                    mh_logor, pair_codes, pairs_of, strata_counts)

coh = load_cohort()
sites = load_sites()

COMPARISONS = {
    # 이름: (검체 필터, 군 열, A, B, 순열 층 열)
    "main_EGFR_vs_KRAS": ("Primary", "grp_main", "EGFR", "KRAS", ["histology", "panel"]),
    "S4_allsamples_EGFR_vs_KRAS": ("all", "grp_main", "EGFR", "KRAS", ["histology", "panel", "sample_type"]),
    "S5_canonical_EGFR_vs_KRAS": ("Primary", "grp_canon", "EGFR", "KRAS", ["histology", "panel"]),
    "aux_STK11KEAP1_vs_neither": ("Primary", "grp_stk", "STK11_KEAP1", "neither", ["histology", "panel"]),
}


def run(name, stype, gcol, A, Bname, strata_cols):
    df = coh if stype == "all" else coh[coh.sample_type == stype]
    df = df[df[gcol].isin([A, Bname])].reset_index(drop=True)
    g = (df[gcol] == A).values
    X = df[[f"site_{s}" for s in sites]].values.astype(int)
    prevA, prevB = X[g].mean(0), X[~g].mean(0)
    elig_sites = [s for k, s in enumerate(sites) if prevA[k] >= GENO_MIN_PREV and prevB[k] >= GENO_MIN_PREV]
    pairs = pairs_of(elig_sites)
    pidx = [(sites.index(a), sites.index(b)) for a, b in pairs]
    codes = pair_codes(X, df["burden"].values.astype(int), pidx)

    def delta(mask):
        wa = mh_logor(strata_counts(codes, mask))
        wb = mh_logor(strata_counts(codes, ~mask))
        return wa, wb

    wa, wb = delta(g)
    d_obs = wa - wb
    rng = np.random.default_rng(SEED)
    strata = df[strata_cols].astype(str).agg("|".join, axis=1).values
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    D = np.empty((N_PERM, len(pairs)))
    for r in range(N_PERM):
        gp = g.copy()
        for idx in groups:
            gp[idx] = g[rng.permutation(idx)]
        a_, b_ = delta(gp)
        D[r] = a_ - b_
    p = (1 + (np.abs(D) >= np.abs(d_obs)[None, :]).sum(0)) / (N_PERM + 1)
    S_obs = np.abs(d_obs).sum()
    S_null = np.abs(D).sum(1)
    out = pd.DataFrame({"comparison": name, "site_i": [a for a, _ in pairs], "site_j": [b for _, b in pairs],
                        "label": [f"{SITE_LABEL[a]}–{SITE_LABEL[b]}" for a, b in pairs],
                        f"w_{A}": wa, f"w_{Bname}": wb, "delta": d_obs, "p_perm": p})
    out["fdr"] = bh(out["p_perm"])
    out.to_csv(RESULTS_DIR / f"geno_edges_{name}.csv", index=False)
    glob = dict(comparison=name, n_A=int(g.sum()), n_B=int((~g).sum()), A=A, B=Bname, eligible_sites=len(elig_sites),
                excluded_sites=";".join(s for s in sites if s not in elig_sites), eligible_pairs=len(pairs),
                strength_A=float(np.abs(wa).sum()), strength_B=float(np.abs(wb).sum()),
                sum_abs_delta=float(S_obs), p_global=float((1 + (S_null >= S_obs).sum()) / (N_PERM + 1)),
                n_fdr10=int((out.fdr < 0.1).sum()), n_fdr05=int((out.fdr < 0.05).sum()),
                min_fdr=float(out.fdr.min()), n_p05=int((out.p_perm < 0.05).sum()),
                median_burden_A=float(df.loc[g, "burden"].median()), median_burden_B=float(df.loc[~g, "burden"].median()))
    prev = pd.DataFrame({"comparison": name, "site": sites, f"prev_{A}": prevA, f"prev_{Bname}": prevB})
    return out, glob, prev, df, g


globs, prevs, pois = [], [], []
for name, spec in COMPARISONS.items():
    out, glob, prev, df, g = run(name, *spec)
    globs.append(glob)
    prevs.append(prev)
    print(f"\n=== {name} ===")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in glob.items()})
    print(out.sort_values("p_perm").head(8).round(3).to_string(index=False))
    # 전이 부담 차이: Poisson GLM + HC0
    dd = df.dropna(subset=["age"]).copy()
    dd["grpA"] = (dd[spec[1]] == spec[2]).astype(int)
    dd["age10"] = dd["age"] / 10
    f = "burden ~ grpA + age10 + C(sex) + C(histology, Treatment('adeno')) + C(panel, Treatment('IMPACT468'))"
    if spec[0] == "all":
        f += " + C(sample_type)"
    for outcome in ("burden", "burden_sel"):
        res = smf.poisson(f.replace("burden ~", f"{outcome} ~"), data=dd).fit(disp=0, cov_type="HC0")
        b, se = res.params["grpA"], res.bse["grpA"]
        pois.append(dict(comparison=name, outcome=outcome, n=len(dd), RR=np.exp(b), lo=np.exp(b - 1.96 * se),
                         hi=np.exp(b + 1.96 * se), p=res.pvalues["grpA"],
                         mean_A=dd.loc[dd.grpA == 1, outcome].mean(), mean_B=dd.loc[dd.grpA == 0, outcome].mean()))

pd.DataFrame(globs).to_csv(RESULTS_DIR / "geno_global.csv", index=False)
pd.concat(prevs).to_csv(RESULTS_DIR / "geno_site_prev.csv", index=False)
pp = pd.DataFrame(pois)
pp.to_csv(RESULTS_DIR / "geno_burden_poisson.csv", index=False)
print(pp.round(3).to_string(index=False))
gl = pd.DataFrame(globs).set_index("comparison")
c3 = gl.loc["main_EGFR_vs_KRAS", "n_fdr10"]
pd.DataFrame([dict(criterion="C3: EGFR vs KRAS Δ간선 FDR<0.1 쌍 >= 1", value=int(c3), met=bool(c3 >= 1))]).to_csv(
    RESULTS_DIR / "criteria_C3.csv", index=False)
print("C3:", c3, c3 >= 1)
