"""3단계: 보조 분석 — (1) curveball 영모형, (2) eLasso Ising 네트워크, (3) 공변량 보정 로지스틱(S2).

출력: results/pairs_curveball.csv, pairs_ising.csv, pairs_S2_logistic.csv, criteria_C1_C2.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

from common import N_CURVEBALL, RESULTS_DIR, SEED, bh, load_cohort, load_sites, pairs_of

warnings.filterwarnings("ignore")
coh = load_cohort()
sites = load_sites()
pairs = pairs_of(sites)
K = len(sites)
X = coh[[f"site_{s}" for s in sites]].values.astype(bool)
n = len(X)
main = pd.read_csv(RESULTS_DIR / "pairs_main.csv")

# ---------- (1) curveball: 행 합(환자별 선택 부위 수)·열 합(부위 유병률) 보존 ----------
rng = np.random.default_rng(SEED)


def trade(M):
    a, b = rng.choice(K, 2, replace=False)
    diff = M[:, a] != M[:, b]
    idx = np.flatnonzero(diff)
    if len(idx) == 0:
        return
    k = int(M[idx, a].sum())
    perm = rng.permutation(idx)
    M[idx, a] = False
    M[idx, b] = True
    M[perm[:k], a] = True
    M[perm[:k], b] = False


def cooc(M):
    Mi = M.astype(np.int32)
    C = Mi.T @ Mi
    return np.array([C[sites.index(a), sites.index(b)] for a, b in pairs])


obs = cooc(X)
M = X.copy()
for _ in range(2000):
    trade(M)
null = np.empty((N_CURVEBALL, len(pairs)))
for r in range(N_CURVEBALL):
    for _ in range(50):
        trade(M)
    null[r] = cooc(M)
assert (M.sum(0) == X.sum(0)).all() and (M.sum(1) == X.sum(1)).all()
mu, sd = null.mean(0), null.std(0, ddof=1)
p = (1 + (np.abs(null - mu) >= np.abs(obs - mu)[None, :]).sum(0)) / (N_CURVEBALL + 1)
cb = pd.DataFrame({"site_i": [a for a, _ in pairs], "site_j": [b for _, b in pairs], "obs": obs,
                   "null_mean": mu, "null_sd": sd, "z": (obs - mu) / sd, "ratio_obs_exp": obs / mu, "p": p})
cb["fdr"] = bh(cb["p"])
cb.to_csv(RESULTS_DIR / "pairs_curveball.csv", index=False)

# ---------- (2) eLasso Ising (EBIC γ=0.25, AND 규칙) ----------
Cgrid = np.logspace(-3, 1, 50)
gamma = 0.25
W = np.zeros((K, K))
for j in range(K):
    y = X[:, j].astype(int)
    Z = np.delete(X, j, axis=1).astype(float)
    best = (np.inf, None)
    for c_val in Cgrid:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=c_val, intercept_scaling=100, max_iter=2000, random_state=SEED)
        m.fit(Z, y)
        pr = np.clip(m.predict_proba(Z)[:, 1], 1e-12, 1 - 1e-12)
        ll = np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
        kk = int((np.abs(m.coef_[0]) > 1e-8).sum())
        ebic = -2 * ll + kk * np.log(n) + 2 * gamma * kk * np.log(K - 1)
        if ebic < best[0]:
            best = (ebic, m.coef_[0].copy())
    W[j, np.arange(K) != j] = best[1]
rows = []
for a, b in pairs:
    i, j = sites.index(a), sites.index(b)
    both = (W[i, j] != 0) and (W[j, i] != 0)
    rows.append(dict(site_i=a, site_j=b, beta_ij=W[i, j], beta_ji=W[j, i],
                     edge_AND=both, weight=(W[i, j] + W[j, i]) / 2 if both else 0.0,
                     sign_agree=np.sign(W[i, j]) == np.sign(W[j, i]) if both else np.nan))
ising = pd.DataFrame(rows)
ising.to_csv(RESULTS_DIR / "pairs_ising.csv", index=False)

# ---------- (3) S2: 공변량 보정 로지스틱 (양방향 적합, 보수적으로 큰 p) ----------
df = coh.dropna(subset=["age"]).copy()
df["age10"] = df["age"] / 10
rows = []
for a, b in pairs:
    df["R"] = np.minimum(df["burden"] - df[f"site_{a}"] - df[f"site_{b}"], 6)
    res = []
    for y_, x_ in ((b, a), (a, b)):
        f = (f"site_{y_} ~ site_{x_} + C(R) + age10 + C(sex) + C(histology, Treatment('adeno'))"
             " + C(panel, Treatment('IMPACT468')) + C(sample_type)")
        try:
            r = smf.logit(f, data=df).fit(disp=0, maxiter=200)
            res.append((r.params[f"site_{x_}"], r.bse[f"site_{x_}"], r.pvalues[f"site_{x_}"]))
        except Exception as e:  # 완전 분리 등
            print("S2 적합 실패", a, b, repr(e)[:80])
            res.append((np.nan, np.nan, np.nan))
    lor = np.nanmean([r[0] for r in res])
    rows.append(dict(site_i=a, site_j=b, n=len(df), logor_ji=res[0][0], logor_ij=res[1][0], logor_mean=lor,
                     or_mean=np.exp(lor), p_max=np.nanmax([r[2] for r in res])))
s2 = pd.DataFrame(rows)
s2["fdr"] = bh(s2["p_max"])
s2.to_csv(RESULTS_DIR / "pairs_S2_logistic.csv", index=False)

# ---------- 성공 기준 C1, C2 ----------
m = main.merge(cb[["site_i", "site_j", "z", "ratio_obs_exp", "fdr"]].rename(columns={"fdr": "cb_fdr"}), on=["site_i", "site_j"])
m = m.merge(ising[["site_i", "site_j", "edge_AND", "weight"]], on=["site_i", "site_j"])
m = m.merge(s2[["site_i", "site_j", "or_mean", "fdr"]].rename(columns={"or_mean": "s2_or", "fdr": "s2_fdr"}), on=["site_i", "site_j"])
m.to_csv(RESULTS_DIR / "pairs_all_methods.csv", index=False)
c1 = m[m.adj_fdr < 0.05]
c2_agree = ((c1.cb_fdr < 0.05) & (np.sign(c1.z) == np.sign(c1.adj_logor))).sum()
ising_agree = ((c1.edge_AND) & (np.sign(c1.weight) == np.sign(c1.adj_logor))).sum()
s2_agree = ((c1.s2_fdr < 0.05) & (np.sign(np.log(c1.s2_or)) == np.sign(c1.adj_logor))).sum()
crit = pd.DataFrame([
    dict(criterion="C1: 부담 보정 MH FDR<0.05 쌍 >= 3", value=len(c1), met=len(c1) >= 3),
    dict(criterion="C2: C1 쌍 중 curveball FDR<0.05·같은 방향 비율 >= 2/3", value=f"{c2_agree}/{len(c1)} = {c2_agree/len(c1):.2f}",
         met=c2_agree / len(c1) >= 2 / 3),
    dict(criterion="(참고) C1 쌍 중 Ising AND 간선·같은 방향", value=f"{ising_agree}/{len(c1)}", met=np.nan),
    dict(criterion="(참고) C1 쌍 중 S2 로지스틱 FDR<0.05·같은 방향", value=f"{s2_agree}/{len(c1)}", met=np.nan),
    dict(criterion="(참고) curveball FDR<0.05 쌍 수 (양/음)", value=f"{int(((cb.fdr<0.05)&(cb.z>0)).sum())}/{int(((cb.fdr<0.05)&(cb.z<0)).sum())}", met=np.nan),
    dict(criterion="(참고) Ising AND 간선 수 (양/음)", value=f"{int((ising.weight>0).sum())}/{int((ising.weight<0).sum())}", met=np.nan),
])
crit.to_csv(RESULTS_DIR / "criteria_C1_C2.csv", index=False)
print(crit.to_string(index=False))
print(m.sort_values("adj_p")[["label", "adj_or", "adj_fdr", "z", "cb_fdr", "weight", "s2_or", "s2_fdr"]].round(3).to_string(index=False))
