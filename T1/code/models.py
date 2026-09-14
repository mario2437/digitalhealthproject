"""모형 적합·예측·도메인 분류기 공통 함수."""
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import N_JOBS, PRED, SEED


def fit_lr(train, vars_=PRED, weights=None):
    X = sm.add_constant(train[vars_].astype(float))
    y = train["y"].astype(float)
    if weights is None:
        return sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return sm.GLM(y, X, family=sm.families.Binomial(), var_weights=np.asarray(weights)).fit()


def pred_lr(m, d, vars_=PRED):
    return np.asarray(m.predict(sm.add_constant(d[vars_].astype(float), has_constant="add")))


def cv_pred_lr(train, vars_=PRED, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    p = np.zeros(len(train))
    for tr, te in skf.split(train, train["y"]):
        m = fit_lr(train.iloc[tr], vars_)
        p[te] = pred_lr(m, train.iloc[te], vars_)
    return p


def _domain_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000))


def domain_auc(ref, tgt, vars_=PRED, kind="lr", k=5):
    """기준자료(0) vs 대상 연도(1) 구분 도메인 분류기의 교차검증 AUROC."""
    X = np.vstack([ref[vars_].values, tgt[vars_].values]).astype(float)
    z = np.r_[np.zeros(len(ref)), np.ones(len(tgt))]
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    s = np.zeros(len(z))
    for tr, te in skf.split(X, z):
        if kind == "lr":
            clf = _domain_lr()
        else:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                     min_child_samples=50, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS,
                                     verbose=-1)
        clf.fit(X[tr], z[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(z, s))


def importance_weights(ref, tgt, vars_=PRED):
    """밀도비 w(x)=P(대상|x)/P(기준|x)×(n기준/n대상), 1·99 백분위 절단, 평균 1 정규화."""
    X = np.vstack([ref[vars_].values, tgt[vars_].values]).astype(float)
    z = np.r_[np.zeros(len(ref)), np.ones(len(tgt))]
    clf = _domain_lr().fit(X, z)
    pr = clf.predict_proba(ref[vars_].values.astype(float))[:, 1]
    w = pr / (1 - pr) * (len(ref) / len(tgt))
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)
    w = w / w.mean()
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    return w, ess


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(b) - np.nanmean(a)) / sp) if sp > 0 else 0.0
