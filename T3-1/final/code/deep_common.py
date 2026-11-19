"""T3-1 심화(deep) 공통: v2 코드(t31_common)를 읽기 전용으로 재사용하고 경로만 deep/로 돌린다."""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEEP = Path(__file__).resolve().parents[1]
V2_CODE = DEEP.parent / "code"  # T3-1/code (v2 파이프라인)
V2_RESULTS = DEEP / "results" / "v2"
os.environ.setdefault("RESULTS_DIR", str(DEEP / "results" / "deep"))
os.environ.setdefault("FIGURES_DIR", str(DEEP / "figures"))
sys.path.insert(0, str(V2_CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

import t31_common as tc  # noqa: E402

RESULTS = Path(os.environ.get("DEEP_RESULTS", DEEP / "results" / "deep"))
FIGURES = DEEP / "figures"
SEED = tc.SEED
OUTCOMES = tc.OUTCOMES
ORGAN_KEYS = tc.ORGAN_KEYS
fast_auc = tc.fast_auc
stratified_auc = tc.stratified_auc
delta_auc_ci = tc.delta_auc_ci
N_BOOT = tc.N_BOOT
MAX_JOBS = int(os.environ.get("MAX_JOBS", 16))  # 결과는 프로세스 수와 무관


def make_lgbm(n_jobs=1):
    p = dict(tc.LGBM_PARAMS)
    p["n_jobs"] = n_jobs
    return LGBMClassifier(**p)


def load_bundle():
    return pd.read_pickle(V2_RESULTS / "features_primary.pkl")


def design(b, spec):
    """spec: 'B', 'B+A', 'B-TMB', 'B+FU', 'B+A+FU'."""
    parts = [b["T"], b["clinical"], b["genomic"]]
    if "+A" in spec:
        parts.append(b["A"])
    if "+FU" in spec:
        parts.append(b["FU"])
    X = pd.concat(parts, axis=1).astype(float)
    if spec == "B-TMB":
        X = X.drop(columns=["TMB_NONSYNONYMOUS"])
    return X


def calib(y, p):
    """시험 세트 보정: O/E, 보정 기울기(y ~ logit p), 보정 절편(기울기 1 고정 offset)."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    oe = p.mean() / y.mean()
    try:
        slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    except Exception:
        slope = np.nan
    try:
        citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        citl = np.nan
    return float(oe), float(slope), float(citl)
