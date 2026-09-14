"""T1(v2) 공통 설정·함수: 경로, 변수 정의, 측정 조화(전환식), 성능·보정 지표.

출력 경로 결정 순서
    1) 명령행 인자 --out DIR  (DIR/results, DIR/figures 사용)
    2) 환경변수 T1_OUT        (동일)
    3) 기본값: 이 파일 기준 ../results, ../figures
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_BOOT = 1000      # 연도별 부트스트랩 반복
N_NULL = 500       # ICI 영점(완전 보정 가정) 모의 반복
N_JOBS = 16

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "KNHANES_SAS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "knhanes" / "sas")))
REF_YEARS = [2016, 2017, 2018]          # 기준(학습) 구간
EVAL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
ALL_YEARS = REF_YEARS + EVAL_YEARS

# 이용지침서(제9기)에 제시된 연구자용 전환식
HBA1C_FROM = 2022   # 2022년 이후 HbA1c: y = 1.034x - 0.0448
BP_FROM = 2021      # 2021년 이후 성인 진동형 혈압계 → 청진형 척도
TG_BEFORE = 2022    # 2005-2021 중성지방: y = 0.99x + 11.94 (새 분석법 척도로)

# 문서화된 변경(그림 주석용)
DOC_CHANGES = {
    2019: "허리둘레 줄자",
    2020: "청진형 혈압계·코로나19",
    2021: "진동형 혈압계·코로나19",
    2022: "검사기관(HbA1c·TG)",
}


def out_dirs(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args(argv)
    root = args.out or os.environ.get("T1_OUT")
    if root:
        res, fig = Path(root) / "results", Path(root) / "figures"
    else:
        res, fig = CODE_DIR.parent / "results", CODE_DIR.parent / "figures"
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return res, fig


# ---------------------------------------------------------------- 예측변수
PRED = ["age", "male", "bmi", "wc", "sbp", "dbp", "pa_aerobic", "sit_hr", "sleep_hr",
        "smk_current", "smk_former", "drk_monthly", "drk_weekly2", "fh_dm"]
VAR_KO = {
    "age": "연령", "male": "남성", "bmi": "BMI", "wc": "허리둘레",
    "sbp": "수축기혈압", "dbp": "이완기혈압", "pa_aerobic": "유산소 신체활동",
    "sit_hr": "좌식시간", "sleep_hr": "수면시간", "smk_current": "현재흡연",
    "smk_former": "과거흡연", "drk_monthly": "음주 월1-4회", "drk_weekly2": "음주 주2회+",
    "fh_dm": "당뇨병 가족력",
}


def hba1c_harmonized(d):
    """조화 척도 HbA1c: 2022년 이후 값에 전환식 적용."""
    h = d["HE_HbA1c"].copy()
    post = d["year"] >= HBA1C_FROM
    h[post] = 1.034 * d.loc[post, "HE_HbA1c"] - 0.0448
    return h


def bp_harmonized(d):
    """2021년 이후(성인, Microlife WatchBP Office) 혈압을 청진형(Greenlight300) 척도로."""
    s, di, a = d["HE_sbp"], d["HE_dbp"], d["age"]
    pp = s - di
    post = d["year"] >= BP_FROM
    sbp, dbp = s.copy(), di.copy()
    sbp[post] = (10.773 + 0.771 * s + 0.039 * a + 0.374 * pp)[post]
    dbp[post] = (13.480 + 0.952 * di - 0.051 * a - 0.098 * pp)[post]
    return sbp, dbp


def tg_harmonized(d):
    """2021년 이전 중성지방을 2022년 새 분석법 척도로."""
    t = d["HE_TG"].copy()
    pre = d["year"] < TG_BEFORE
    t[pre] = 0.99 * d.loc[pre, "HE_TG"] + 11.94
    return t


def scenario(d, hba1c_conv=False, bp_conv=False, outcome="composite"):
    """코호트(01 산출물)에 결과변수·예측변수 척도를 시나리오에 맞게 설정."""
    o = d.copy()
    o["hba1c"] = hba1c_harmonized(o) if hba1c_conv else o["HE_HbA1c"]
    if bp_conv:
        o["sbp"], o["dbp"] = bp_harmonized(o)
    else:
        o["sbp"], o["dbp"] = o["HE_sbp"], o["HE_dbp"]
    fpg = (o["HE_glu"] >= 100) & (o["HE_glu"] <= 125)
    a1c = (o["hba1c"] >= 5.7) & (o["hba1c"] <= 6.4)
    o["y"] = {"composite": fpg | a1c, "fpg": fpg, "hba1c": a1c}[outcome].astype(int)
    return o


# ---------------------------------------------------------------- 지표
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def ici(y, p):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    # delta=0.005: 예측확률 0.005 이내 점은 선형보간(계산 가속; 값 차이는 소수 넷째 자리 이하)
    sm_ = lowess(y, p, frac=2 / 3, it=0, delta=0.005, return_sorted=False)
    return float(np.mean(np.abs(sm_ - p)))


def citl(y, p):
    """보정 절편(calibration-in-the-large): 기울기 1 고정 offset 로지스틱."""
    # 절편 하나만 추정하므로 뉴턴법으로 직접 푼다(statsmodels GLM offset 적합과 같은 해)
    y = np.asarray(y, float)
    lp = logit(np.asarray(p, float))
    a = 0.0
    for _ in range(50):
        mu = expit(lp + a)
        step = np.sum(y - mu) / np.sum(mu * (1 - mu))
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def cal_slope(y, p):
    import statsmodels.api as sm
    m = sm.GLM(y, sm.add_constant(logit(p)), family=sm.families.Binomial()).fit()
    return float(m.params[1])


def mbc(p):
    """모형기반 기대 AUROC(van Klaveren 등 2016). 동점은 1/2."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ps = p[order]
    q = 1 - ps
    cq = np.concatenate([[0.0], np.cumsum(q)])
    lo = np.searchsorted(ps, ps, side="left")
    hi = np.searchsorted(ps, ps, side="right")
    less = cq[lo]                       # p_j < p_i 인 j의 (1-p_j) 합
    ties = cq[hi] - cq[lo] - q          # 자기 자신 제외 동점
    num = np.sum(ps * (less + 0.5 * ties))
    den = np.sum(ps * (cq[-1] - q))
    return float(num / den)


def point_metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return dict(n=len(y), prev=float(y.mean()), mean_pred=float(p.mean()),
                OE=float(y.mean() / p.mean()), AUROC=float(roc_auc_score(y, p)),
                AUPRC=float(average_precision_score(y, p)), mbc=mbc(p),
                CITL=citl(y, p), slope=cal_slope(y, p), ICI=ici(y, p))


def ici_null(p, n_sim=N_NULL, seed=SEED):
    """완전 보정(y~Bernoulli(p)) 가정에서의 ICI 분포: 표본크기에 따른 우연 오차 수준."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p)
    vals = np.array([ici(rng.binomial(1, p), p) for _ in range(n_sim)])
    return float(np.median(vals)), float(np.percentile(vals, 95))


def boot_indices(n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def pct_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
