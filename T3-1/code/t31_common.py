"""T3-1 v2 공통 설정: 경로, 시드, 모델, 교차검증, 평가 함수, 특징 묶음 조립."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("RESULTS_DIR", HERE / "../results")).resolve()
FIGURES = Path(os.environ.get("FIGURES_DIR", HERE / "../figures")).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20260914
N_FOLDS = 5
N_BOOT = int(os.environ.get("N_BOOT", 2000))
N_JOBS = min(16, os.cpu_count() or 1)
MIN_FREQ = 0.01          # 유전체 특징 빈도 필터 (코호트 내 1% 이상)
BURDEN_CUT = 3           # HIGH_BURDEN = MET_SITE_COUNT >= 3 (SAP 고정)
TRAIN_FRAC = 0.70        # 시간 분할: PID_NUM 앞 70% 학습

LGBM_PARAMS = dict(  # 사전 고정 (튜닝하지 않음, v1과 동일)
    n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=N_JOBS, verbose=-1,
)

OUTCOMES = {  # 5개 장기 + 전이 부담
    "DMETS_DX_LUNG": "폐", "DMETS_DX_LIVER": "간", "DMETS_DX_BONE": "뼈",
    "DMETS_DX_CNS_BRAIN": "뇌", "DMETS_DX_INTRA_ABDOMINAL": "복강내", "HIGH_BURDEN": "부담≥3",
}
ORGAN_KEYS = list(OUTCOMES)[:5]


def make_lgbm():
    return LGBMClassifier(**LGBM_PARAMS)


def design(bundle: dict, spec: str, drop_panel: bool = False) -> pd.DataFrame:
    """spec: 'T', 'A', 'T+A', 'B', 'B+A' (B는 T를 포함)."""
    parts = []
    if spec in ("T", "T+A", "B", "B+A"):
        parts.append(bundle["T"])
    if spec in ("B", "B+A"):
        parts += [bundle["clinical"], bundle["genomic"]]
    if spec in ("A", "T+A", "B+A"):
        A = bundle["A"]
        if drop_panel:
            A = A[[c for c in A.columns if not c.startswith("PANEL_")]]
        parts.append(A)
    return pd.concat(parts, axis=1).astype(float)


def oof_predict(X: pd.DataFrame, y: np.ndarray, factory=make_lgbm, seed=SEED) -> np.ndarray:
    """층화 5-겹 교차검증 out-of-fold 예측 확률."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    Xv = X.values.astype(float)
    for tr, te in skf.split(Xv, y):
        oof[te] = factory().fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
    return oof


def fast_auc(y, s):
    """순위 기반 AUROC (동점 평균 순위)."""
    y = np.asarray(y)
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stratified_auc(y, s, strata):
    """층 내부 AUROC의 표본 수 가중 평균 (사건/비사건이 모두 있는 층만)."""
    y, s, strata = np.asarray(y), np.asarray(s), np.asarray(strata)
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        if y[m].min() == y[m].max():
            continue
        num += m.sum() * fast_auc(y[m], s[m])
        den += m.sum()
    return num / den


def boot_idx(n, rng):
    return rng.integers(0, n, n)


def paired_bootstrap(stat, n, n_boot=N_BOOT, seed=SEED):
    """stat(idx) -> float 을 부트스트랩. 반환 (2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        try:
            v = stat(boot_idx(n, rng))
        except (ZeroDivisionError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def delta_auc_ci(y, s_new, s_ref, strata=None, n_boot=N_BOOT, seed=SEED):
    """AUROC(new) − AUROC(ref) 점추정과 쌍체 부트스트랩 95% CI. strata가 있으면 층화 AUROC."""
    y = np.asarray(y)
    f = (lambda yy, ss, gg: stratified_auc(yy, ss, gg)) if strata is not None else (lambda yy, ss, gg: fast_auc(yy, ss))
    g = np.asarray(strata) if strata is not None else np.zeros(len(y))

    def stat(i):
        if y[i].min() == y[i].max():
            return np.nan
        return f(y[i], s_new[i], g[i]) - f(y[i], s_ref[i], g[i])

    point = f(y, s_new, g) - f(y, s_ref, g)
    lo, hi = paired_bootstrap(stat, len(y), n_boot, seed)
    return point, lo, hi
