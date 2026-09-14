"""T3-2(v2) 공통 설정·함수: 경로, 전이 부위(노드), Mantel–Haenszel 계산, FDR.

경로는 환경변수로 바꿀 수 있다.
    T32_DATA_DIR     (기본 <저장소>/data/msk_met_2021)
    T32_RESULTS_DIR  (기본 ../results  — code/ 기준)
    T32_FIGURES_DIR  (기본 ../figures)
"""
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 2026
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("T32_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "msk_met_2021")))
RESULTS_DIR = Path(os.environ.get("T32_RESULTS_DIR", CODE_DIR.parent / "results")).resolve()
FIGURES_DIR = Path(os.environ.get("T32_FIGURES_DIR", CODE_DIR.parent / "figures")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---- SAP 3절: 노드 선택 규칙 ----
MIN_PREV = 0.03          # 네트워크 코호트 유병률 3% 이상
STRATA_CAP = 6           # 나머지 부담 층: 0..5, >=6
GENO_MIN_PREV = 0.05     # 유전형 비교: 두 군 모두 유병률 5% 이상인 부위의 쌍만
N_CURVEBALL = 1000
N_PERM = 2000
EXCLUDE_MUT_TYPES = {"5'Flank"}  # 비침묵 변이 = 5'Flank 제외 전부 (v1 과 동일)

# 부위 코드 → 그림용 짧은 한국어 이름 (흉강 내 부위를 앞에 둔다)
SITE_LABEL = {
    "LUNG": "폐(폐내)", "PLEURA": "흉막", "MEDIASTINUM": "종격동",
    "CNS_BRAIN": "뇌", "BONE": "뼈", "LIVER": "간", "ADRENAL_GLAND": "부신",
    "DIST_LN": "원격 림프절", "PNS": "말초신경계", "INTRA_ABDOMINAL": "복강 내",
    "BILIARY_TRACT": "담도", "KIDNEY": "신장",
    "SKIN": "피부", "BOWEL": "장", "HEAD_NECK": "두경부", "BLADDER_UT": "방광·요로",
    "FEMALE_GENITAL": "여성생식기", "BREAST": "유방", "OVARY": "난소", "MALE_GENITAL": "남성생식기",
    "UNSPECIFIED": "부위 불명",
}
THORACIC = {"LUNG", "PLEURA", "MEDIASTINUM"}


def read_tsv(name, **kw):
    return pd.read_csv(DATA_DIR / name, sep="\t", comment="#", low_memory=False, **kw)


def load_cohort():
    return pd.read_csv(RESULTS_DIR / "cohort.csv")


def load_sites():
    return pd.read_csv(RESULTS_DIR / "site_selection.csv").query("selected")["site"].tolist()


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def pairs_of(sites):
    return list(combinations(sites, 2))


# ---------------- 2x2 / Mantel–Haenszel ----------------
def pair_codes(X, B, pairs_idx, cap=STRATA_CAP):
    """X: (n, K) 0/1, B: (n,) 전체 부담. 각 쌍에 대해 (층*4 + 칸) 코드 (n, P).
    칸: 3=(i1,j1) 2=(i1,j0) 1=(i0,j1) 0=(i0,j0). 층 = min(B - Xi - Xj, cap)."""
    ii = np.array([a for a, _ in pairs_idx])
    jj = np.array([b for _, b in pairs_idx])
    Xi, Xj = X[:, ii], X[:, jj]
    R = np.minimum(B[:, None] - Xi - Xj, cap)
    assert (R >= 0).all(), "부담이 선택 부위 수보다 작음"
    return (R * 4 + 2 * Xi + Xj).astype(np.int64)


def strata_counts(codes, rows=None, cap=STRATA_CAP):
    """codes (n, P) → counts (P, S, 4)."""
    n_str = cap + 1
    c = codes if rows is None else codes[rows]
    P = c.shape[1]
    flat = (c + (np.arange(P) * n_str * 4)[None, :]).ravel()
    return np.bincount(flat, minlength=P * n_str * 4).reshape(P, n_str, 4).astype(float)


def mh_logor(counts):
    """counts (P, S, 4) → MH log OR (P,). 분자·분모가 0이면 관찰된 층의 모든 칸 +0.5."""
    a, b, c, d = counts[..., 3], counts[..., 2], counts[..., 1], counts[..., 0]
    n = counts.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.where(n > 0, a * d / n, 0).sum(-1)
        den = np.where(n > 0, b * c / n, 0).sum(-1)
        bad = (num == 0) | (den == 0)
        if bad.any():
            obs = (n > 0)[bad][..., None]
            cc = counts[bad] + 0.5 * obs
            a2, b2, c2, d2 = cc[..., 3], cc[..., 2], cc[..., 1], cc[..., 0]
            n2 = cc.sum(-1)
            num[bad] = np.where(n2 > 0, a2 * d2 / n2, 0).sum(-1)
            den[bad] = np.where(n2 > 0, b2 * c2 / n2, 0).sum(-1)
        return np.log(num / den)


def cmh_test(counts):
    """Cochran–Mantel–Haenszel 검정 (연속성 보정 없음) → (chi2, p). counts (P, S, 4)."""
    a, b, c, d = counts[..., 3], counts[..., 2], counts[..., 1], counts[..., 0]
    n = counts.sum(-1)
    r1, r0, c1, c0 = a + b, c + d, a + c, b + d
    with np.errstate(divide="ignore", invalid="ignore"):
        ea = np.where(n > 1, r1 * c1 / n, 0)
        va = np.where(n > 1, r1 * r0 * c1 * c0 / (n ** 2 * (n - 1)), 0)
        aa = np.where(n > 1, a, 0)
        chi2 = (aa.sum(-1) - ea.sum(-1)) ** 2 / va.sum(-1)
    return chi2, stats.chi2.sf(chi2, 1)


def mh_ci(counts):
    """Robins–Breslow–Greenland 분산으로 MH log OR 95% CI. counts (P, S, 4)."""
    a, b, c, d = counts[..., 3], counts[..., 2], counts[..., 1], counts[..., 0]
    n = counts.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nz = n > 0
        P_ = np.where(nz, (a + d) / n, 0); Q_ = np.where(nz, (b + c) / n, 0)
        R_ = np.where(nz, a * d / n, 0); S_ = np.where(nz, b * c / n, 0)
        Rs, Ss = R_.sum(-1), S_.sum(-1)
        var = ((P_ * R_).sum(-1) / (2 * Rs ** 2) + ((P_ * S_ + Q_ * R_)).sum(-1) / (2 * Rs * Ss)
               + (Q_ * S_).sum(-1) / (2 * Ss ** 2))
    se = np.sqrt(var)
    lo = np.log(Rs / Ss) - 1.96 * se
    hi = np.log(Rs / Ss) + 1.96 * se
    return se, lo, hi
