"""D1/D2 계산: 해결 전략(S0–S7) × 결과 × 시간 분할 컷, 내부 평가 방식(E0–E3)과 시험 성능.

SAP_addendum.md D1·D2 구현. 과제 단위 병렬(joblib 프로세스 ≤16, LightGBM n_jobs=1).
출력: results/d1_metrics.tsv, results/d1_test_pred_c70.tsv.gz
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from deep_common import (MAX_JOBS, OUTCOMES, RESULTS, SEED, calib, design, fast_auc, load_bundle, make_lgbm,
                         stratified_auc)

K_MARG = 30
b = load_bundle()
N = len(b["Y"])
PID = b["A"]["PID_NUM"].values
PANEL = b["meta"]["GENE_PANEL"].values
OS = np.expm1(b["FU"]["LOG_OS_MONTHS"].values)
DEAD = b["FU"]["OS_DECEASED"].values
COMPLETE24 = (OS >= 24) | (DEAD == 1)
COMPLETE36 = (OS >= 36) | (DEAD == 1)
A_COLS = list(b["A"].columns)
FU_COLS = list(b["FU"].columns)
NON_T = list(b["clinical"].columns) + list(b["genomic"].columns)
q = lambda c: np.quantile(PID, c)
CUTS = {  # 이름: (학습 마스크, 시험 마스크)
    "c70": (PID <= q(.70), PID > q(.70)),
    "c50": (PID <= q(.50), (PID > q(.50)) & (PID <= q(.70))),
    "c60": (PID <= q(.60), (PID > q(.60)) & (PID <= q(.80))),
    "c80": (PID <= q(.80), PID > q(.80)),
}
STRATS = {  # 코드: (spec, 설명)
    "S0": ("B+A", "흔적 포함"), "S1": ("B", "흔적 제거"), "S1b": ("B-TMB", "엄격 제거(TMB 제외)"),
    "S2": ("B", "라벨 완전 근사 학습(24개월)"), "S2b": ("B", "라벨 완전 근사 학습(36개월)"),
    "S3": ("B", "패널 균형 가중"), "S4": ("B", "시기 잔차화"),
    "S5a": ("B+A", "흔적 고정값 예측"), "S5b": ("B+A", "흔적 주변화 예측"),
    "S6": ("B+FU", "추적 주변화 예측"), "S7": ("B+A+FU", "흔적·추적 주변화 예측"),
}
XS = {spec: design(b, spec) for spec in {s for s, _ in STRATS.values()}}


def predictor(code, y, tr, ev, seed=SEED):
    """전역 인덱스 tr로 학습, ev 예측. 전략별 변환 포함."""
    spec = STRATS[code][0]
    X = XS[spec]
    keep = X.iloc[tr].nunique().values > 1
    cols = X.columns[keep]
    Xtr = X.iloc[tr][cols].copy()
    Xev = X.iloc[ev][cols].copy()
    w = None
    if code == "S3":
        freq = pd.Series(PANEL[tr]).value_counts(normalize=True)
        w = 1.0 / pd.Series(PANEL[tr]).map(freq).values
        w = w / w.mean()
    if code == "S4":
        edges = np.quantile(PID[tr], [.2, .4, .6, .8])
        bins = np.searchsorted(edges, PID[tr], side="left")
        nt = [c for c in NON_T if c in cols]
        means = Xtr[nt].groupby(bins).mean()
        Xtr[nt] = Xtr[nt].values - means.loc[bins].values
        Xev[nt] = Xev[nt].values - means.loc[4].values
    model = make_lgbm().fit(Xtr.values, y[tr], sample_weight=w)
    if code == "S5a":
        a_tr = b["A"].iloc[tr]
        last = PID[tr] > np.quantile(PID[tr], .9)
        ref = {"PANEL_IMPACT341": 0, "PANEL_IMPACT410": 0, "PANEL_IMPACT468": 1, "PID_NUM": PID[tr].max(),
               "SAMPLE_ORDER": 1}
        for c in ["TUMOR_PURITY", "SAMPLE_COVERAGE", "SEQ_LAG_YEARS"]:
            ref[c] = float(a_tr[c][last].median())
        for c, v in ref.items():
            if c in Xev.columns:
                Xev[c] = v
        return model.predict_proba(Xev.values)[:, 1]
    if code in ("S5b", "S6", "S7"):
        repl = [c for c in (A_COLS if code == "S5b" else FU_COLS if code == "S6" else A_COLS + FU_COLS) if c in cols]
        rng = np.random.default_rng(seed)
        draws = rng.choice(tr, K_MARG, replace=False)
        src = X.iloc[draws][repl].values
        Xv = Xev.values.copy()
        pos = [list(cols).index(c) for c in repl]
        out = np.zeros(len(ev))
        for k in range(K_MARG):
            Xv[:, pos] = src[k]
            out += model.predict_proba(Xv)[:, 1]
        return out / K_MARG
    return model.predict_proba(Xev.values)[:, 1]


def run_task(cut, o, code):
    t0 = time.time()
    y = b["Y"][o].values
    trm, tem = CUTS[cut]
    pool = np.where(trm & (COMPLETE24 if code == "S2" else COMPLETE36 if code == "S2b" else True))[0]
    test = np.where(tem)[0]
    r = dict(cut=cut, outcome=o, strategy=code, n_pool=len(pool), n_test=len(test),
             prev_pool=y[pool].mean(), prev_test=y[test].mean())
    # E0 무작위 5-겹 (+E1 패널 층화)
    oof = np.zeros(len(pool))
    for a, v in StratifiedKFold(5, shuffle=True, random_state=SEED).split(pool, y[pool]):
        oof[v] = predictor(code, y, pool[a], pool[v])
    r["auc_E0"] = fast_auc(y[pool], oof)
    r["oe_E0"] = oof.mean() / y[pool].mean()
    r["auc_E1"] = stratified_auc(y[pool], oof, PANEL[pool])
    # E2 시기 구간 그룹 CV (S0·S1)
    if code in ("S0", "S1"):
        edges = np.quantile(PID[pool], [.2, .4, .6, .8])
        g = np.searchsorted(edges, PID[pool], side="left")
        oof2 = np.zeros(len(pool))
        for k in range(5):
            oof2[g == k] = predictor(code, y, pool[g != k], pool[g == k])
        r["auc_E2"] = fast_auc(y[pool], oof2)
    # E3 전향 홀드아웃 (학습 안 PID 앞 70% → 뒤 30%)
    order = pool[np.argsort(PID[pool], kind="stable")]
    k70 = int(round(0.7 * len(order)))
    p3 = predictor(code, y, order[:k70], order[k70:])
    r["auc_E3"] = fast_auc(y[order[k70:]], p3)
    r["oe_E3"] = p3.mean() / y[order[k70:]].mean()
    # 외부(시험)
    pt = predictor(code, y, pool, test)
    r["auc_test"] = fast_auc(y[test], pt)
    r["oe_test"], r["cal_slope"], r["cal_int"] = calib(y[test], pt)
    cm = COMPLETE24[test]
    r["n_test_complete"] = int(cm.sum())
    r["prev_test_complete"] = y[test][cm].mean()
    r["auc_test_complete"] = fast_auc(y[test][cm], pt[cm])
    r["oe_test_complete"] = pt[cm].mean() / y[test][cm].mean()
    r["sec"] = round(time.time() - t0, 1)
    return r, (pt if cut == "c70" else None)


if __name__ == "__main__":
    tasks = [(c, o, s) for c in CUTS for o in OUTCOMES for s in STRATS]
    print("tasks", len(tasks), flush=True)
    import d1_strategies as mod  # 작업자 프로세스가 모듈을 import 하도록(큰 전역 객체 직렬화 방지)
    res = Parallel(n_jobs=MAX_JOBS, verbose=5)(delayed(mod.run_task)(*t) for t in tasks)
    m = pd.DataFrame([r for r, _ in res])
    m["gap_E0"] = m.auc_E0 - m.auc_test
    m.to_csv(RESULTS / "d1_metrics.tsv", sep="\t", index=False, float_format="%.5f")
    te_idx = np.where(CUTS["c70"][1])[0]
    pred = pd.DataFrame({"SAMPLE_ID": b["meta"]["SAMPLE_ID"].values[te_idx]})
    for (c, o, s), (_, p) in zip(tasks, res):
        if p is not None:
            pred[f"{o}|{s}"] = p
    pred.to_csv(RESULTS / "d1_test_pred_c70.tsv.gz", sep="\t", index=False, float_format="%.6f")
    print(m[m.cut == "c70"].pivot_table(index="strategy", columns="outcome", values="auc_test").round(3))
    print("done")
