"""D3: 중요도 방법 교차검증 (SAP_addendum D3).

B+A + 영 특징(NULL_NORMAL, NULL_RANKINT, NULL_BINARY):
 (i) 전체 적합 TreeSHAP 평균 |SHAP| 순위
 (ii) 5-겹 교차검증 퍼뮤테이션 ΔAUROC (개별·묶음, 5회 반복 평균, 쌍체 부트스트랩 CI)
 (iii) 주 시간 분할(c70) 외부 퍼뮤테이션 ΔAUROC
출력: results/d3_shap_ranks.tsv, d3_perm_cv.tsv, d3_perm_external.tsv, d3_summary.json
"""
import json

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from deep_common import (MAX_JOBS, N_BOOT, ORGAN_KEYS, OUTCOMES, RESULTS, SEED, design, fast_auc, load_bundle,
                         make_lgbm)

R = 5
b = load_bundle()
N = len(b["Y"])
rng = np.random.default_rng(SEED)
X = design(b, "B+A")
X["NULL_NORMAL"] = rng.standard_normal(N)
X["NULL_RANKINT"] = rng.permutation(N).astype(float) + 1
X["NULL_BINARY"] = (rng.random(N) < 0.10).astype(float)
cols = list(X.columns)
A = list(b["A"].columns)
GROUPS = {
    "A_시기": [c for c in A if c.startswith("PANEL_")] + ["PID_NUM"],
    "A_품질": ["SAMPLE_ORDER", "TUMOR_PURITY", "SAMPLE_COVERAGE", "SEQ_LAG_YEARS"],
    "A_전체": A, "T": list(b["T"].columns), "C": list(b["clinical"].columns), "G": list(b["genomic"].columns),
}
SINGLES = A + ["NULL_NORMAL", "NULL_RANKINT", "NULL_BINARY"]
ITEMS = {**{f: [f] for f in SINGLES}, **GROUPS}
ITEMS_EXT = {k: ITEMS[k] for k in ["PID_NUM", "NULL_NORMAL", "NULL_RANKINT", "NULL_BINARY",
                                   "A_시기", "A_품질", "A_전체", "T", "C", "G"]}
Xv = X.values
PID = b["A"]["PID_NUM"].values
cut = np.quantile(PID, 0.70)
TR, TE = np.where(PID <= cut)[0], np.where(PID > cut)[0]


def permuted_preds(model, Xe, items, seed):
    """Xe 안에서 묶음의 행을 함께 섞어 R회 예측."""
    r = np.random.default_rng(seed)
    out = {}
    for name, cs in items.items():
        pos = [cols.index(c) for c in cs]
        ps = []
        for _ in range(R):
            Xp = Xe.copy()
            Xp[:, pos] = Xe[r.permutation(len(Xe))][:, pos]
            ps.append(model.predict_proba(Xp)[:, 1])
        out[name] = np.vstack(ps)
    return out


def perm_ci(y, s0, P, seed):
    base = fast_auc(y, s0)
    point = float(np.mean([base - fast_auc(y, p) for p in P]))
    r = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        i = r.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a0 = fast_auc(y[i], s0[i])
        vals.append(np.mean([a0 - fast_auc(y[i], p[i]) for p in P]))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    shap_rows, cv_jobs, ext_jobs = [], [], []
    for o in OUTCOMES:
        y = b["Y"][o].values
        # (i) SHAP
        m = make_lgbm(MAX_JOBS).fit(Xv, y)
        sv = shap.TreeExplainer(m).shap_values(Xv)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(imp) + 1), index=imp.index)
        for f in SINGLES:
            shap_rows.append(dict(outcome=o, 결과=OUTCOMES[o], feature=f, shap_rank=int(rank[f]),
                                  mean_abs_shap=float(imp[f])))
        for g, cs in GROUPS.items():
            shap_rows.append(dict(outcome=o, 결과=OUTCOMES[o], feature=g, shap_rank=np.nan,
                                  mean_abs_shap=float(imp[cs].sum())))
        # (ii) CV 퍼뮤테이션
        oof = np.zeros(N)
        P = {k: np.zeros((R, N)) for k in ITEMS}
        for f_i, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xv, y)):
            mf = make_lgbm(MAX_JOBS).fit(Xv[a], y[a])
            oof[v] = mf.predict_proba(Xv[v])[:, 1]
            for k, arr in permuted_preds(mf, Xv[v], ITEMS, SEED + f_i).items():
                P[k][:, v] = arr
        for k in ITEMS:
            cv_jobs.append((o, k, y, oof.copy(), P[k]))
        # (iii) 외부
        me = make_lgbm(MAX_JOBS).fit(Xv[TR], y[TR])
        st = me.predict_proba(Xv[TE])[:, 1]
        for k, arr in permuted_preds(me, Xv[TE], ITEMS_EXT, SEED + 99).items():
            ext_jobs.append((o, k, y[TE], st, arr))
        print("fit done", o, flush=True)

    pd.DataFrame(shap_rows).to_csv(RESULTS / "d3_shap_ranks.tsv", sep="\t", index=False, float_format="%.5f")
    out = {}
    for name, jobs in [("cv", cv_jobs), ("external", ext_jobs)]:
        res = Parallel(n_jobs=MAX_JOBS)(delayed(perm_ci)(y, s0, P, SEED) for _, _, y, s0, P in jobs)
        df = pd.DataFrame([dict(outcome=o, 결과=OUTCOMES[o], item=k, delta_auc=p, ci_lo=lo, ci_hi=hi)
                           for (o, k, *_), (p, lo, hi) in zip(jobs, res)])
        df.to_csv(RESULTS / f"d3_perm_{name}.tsv", sep="\t", index=False, float_format="%.5f")
        out[name] = df

    cv = out["cv"].set_index(["outcome", "item"])
    hd7 = []
    for o in ORGAN_KEYS:
        pidr = cv.loc[(o, "PID_NUM")]
        null_hi = max(cv.loc[(o, n), "ci_hi"] for n in ["NULL_NORMAL", "NULL_RANKINT", "NULL_BINARY"])
        hd7.append(bool(pidr.ci_lo > 0 and pidr.delta_auc > null_hi))
    ext = out["external"].set_index(["outcome", "item"])
    hd8 = [bool(ext.loc[(o, "A_시기"), "delta_auc"] <= 0.005) for o in ORGAN_KEYS]
    sr = pd.DataFrame(shap_rows)
    summary = {
        "HD7 충족 장기(PID 퍼뮤테이션 CI>0 & 영 특징 상한 초과)": dict(zip(ORGAN_KEYS, hd7)),
        "HD7 판정(>=2/5)": sum(hd7) >= 2,
        "HD8 충족 장기(외부 A_시기 ΔAUROC<=0.005)": dict(zip(ORGAN_KEYS, hd8)),
        "HD8 판정(>=4/5)": sum(hd8) >= 4,
        "영 특징 SHAP 순위": sr[sr.feature.str.startswith("NULL")].pivot(index="outcome", columns="feature",
                                                                   values="shap_rank").to_dict(orient="index"),
        "N 특징": len(cols),
    }
    (RESULTS / "d3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=int))
    print(out["cv"].pivot(index="item", columns="결과", values="delta_auc").round(4).to_string())
    print(out["external"].pivot(index="item", columns="결과", values="delta_auc").round(4).to_string())
