"""04. 보정 오차의 원인 분해(주 결과: 당뇨병 전단계 composite).

사전 지정 순서(필요한 정보가 적은 것부터)
  S0 원자료 → S1 측정 조화(공식 전환식: 2021+ 혈압, 2022+ HbA1c)
  → S2 공변량 이동 보정(도메인 분류기 밀도비 가중 재적합; 라벨 없는 대상 자료만 사용)
  → S3 사전확률 이동 보정(대상 연도 결과로 절편만 재추정; 오라클)
초과 보정오차 E_k = ICI(S_k) - ICI_null(S_k 예측확률의 완전보정 모의 중앙값)
원인별 몫 = 단계별 E 감소량 / E_0  (E_0 < 0.01 이면 계산하지 않음)
추가: 순서 민감도(공변량 보정을 측정 조화보다 먼저), 유병률 변화 분해(Oaxaca 형), 원자료 척도 코호트 민감도
산출: results/decomposition.csv, decomposition_shares.csv(순서 민감도 열 포함), prevalence_decomp.csv, naive_cohort.csv
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (EVAL_YEARS, N_JOBS, REF_YEARS, SEED, boot_indices, cal_slope, citl, expit,
                    ici, ici_null, logit, out_dirs, pct_ci, scenario)
from models import fit_lr, importance_weights, pred_lr

RES, _ = out_dirs()
MIN_E0 = 0.01


def year_job(year, ref, t0, t1):
    m0 = fit_lr(ref)
    y0, p0 = t0["y"].values, pred_lr(m0, t0)
    y1, p1 = t1["y"].values, pred_lr(m0, t1)
    w, ess = importance_weights(ref, t1)
    p2 = pred_lr(fit_lr(ref, weights=w), t1)
    a3 = citl(y1, p2)
    p3 = expit(logit(p2) + a3)
    # 순서 민감도: 원자료 척도에서 공변량 보정 먼저
    w0, _ = importance_weights(ref, t0)
    p0w = pred_lr(fit_lr(ref, weights=w0), t0)

    P = {"S0": (y0, p0), "S1": (y1, p1), "S2": (y1, p2), "S3": (y1, p3), "S0w": (y0, p0w)}
    nulls = {k: ici_null(p, seed=SEED + year)[0] for k, (_, p) in P.items()}
    pts = {k: ici(y, p) for k, (y, p) in P.items()}

    def shares(I):
        E = {k: I[k] - nulls[k] for k in I}
        if E["S0"] < MIN_E0:
            return dict(E0=E["S0"], sh_meas=np.nan, sh_cov=np.nan, sh_prior=np.nan, sh_resid=np.nan,
                        sh_cov_first=np.nan, sh_meas_second=np.nan)
        return dict(E0=E["S0"], sh_meas=(E["S0"] - E["S1"]) / E["S0"],
                    sh_cov=(E["S1"] - E["S2"]) / E["S0"], sh_prior=(E["S2"] - E["S3"]) / E["S0"],
                    sh_resid=E["S3"] / E["S0"],
                    sh_cov_first=(E["S0"] - E["S0w"]) / E["S0"],
                    sh_meas_second=(E["S0w"] - E["S2"]) / E["S0"])

    point = shares(pts)
    boots = []
    for idx in boot_indices(len(y0), seed=SEED + year):
        I = {k: ici(y[idx], p[idx]) for k, (y, p) in P.items() if k != "S3"}
        ab = citl(y1[idx], p2[idx])
        I["S3"] = ici(y1[idx], expit(logit(p2[idx]) + ab))
        boots.append(dict(**{f"ICI_{k}": v for k, v in I.items()}, **shares(I)))
    B = pd.DataFrame(boots)

    rows = []
    for k, (y, p) in P.items():
        lo, hi = pct_ci(B[f"ICI_{k}"])
        rows.append(dict(year=year, scenario=k, n=len(y), prev=y.mean(), mean_pred=p.mean(),
                         OE=y.mean() / p.mean(), CITL=citl(y, p), slope=cal_slope(y, p),
                         ICI=pts[k], ICI_lo=lo, ICI_hi=hi, ICI_null=nulls[k],
                         excess=pts[k] - nulls[k], ess_weights=ess if k == "S2" else np.nan,
                         alpha_S3=a3 if k == "S3" else np.nan))
    sh = dict(year=year, **point)
    for c in ["sh_meas", "sh_cov", "sh_prior", "sh_resid", "sh_cov_first", "sh_meas_second"]:
        sh[f"{c}_lo"], sh[f"{c}_hi"] = pct_ci(B[c])
        sh[f"{c}_nboot_valid"] = int(B[c].notna().sum())
    prev_ref = ref["y"].mean()
    ox = [dict(year=year, scale=s, prev_ref=prev_ref, prev_year=y.mean(), mean_pred=p.mean(),
               delta_prev=y.mean() - prev_ref, explained_covariates=p.mean() - prev_ref,
               unexplained_conditional=y.mean() - p.mean())
          for s, (y, p) in (("S0", P["S0"]), ("S1", P["S1"]))]
    return rows, sh, ox


def main():
    base = pd.read_pickle(RES / "base_cohort.pkl")
    fixed = base[~base["undx_harm"] & base["cc"]].reset_index(drop=True)
    d0, d1 = scenario(fixed), scenario(fixed, hba1c_conv=True, bp_conv=True)
    ref = d0[d0["year"].isin(REF_YEARS)]
    res = Parallel(n_jobs=min(N_JOBS, len(EVAL_YEARS)))(
        delayed(year_job)(y, ref, d0[d0["year"] == y], d1[d1["year"] == y]) for y in EVAL_YEARS)
    dec = pd.DataFrame([r for rows, _, _ in res for r in rows])
    sh = pd.DataFrame([s for _, s, _ in res])
    ox = pd.DataFrame([o for _, _, oo in res for o in oo])
    dec.to_csv(RES / "decomposition.csv", index=False)
    sh.to_csv(RES / "decomposition_shares.csv", index=False)
    ox.to_csv(RES / "prevalence_decomp.csv", index=False)
    pd.set_option("display.width", 250)
    print(dec.round(3).to_string())
    print(sh[["year", "E0", "sh_meas", "sh_meas_lo", "sh_meas_hi", "sh_cov", "sh_cov_lo", "sh_cov_hi",
              "sh_prior", "sh_resid", "sh_cov_first", "sh_meas_second"]].round(3).to_string())
    print(ox.round(3).to_string())

    # 원자료 척도 HbA1c로 미진단 당뇨병을 판정한 코호트(민감도)
    naive = base[~base["undx_raw"] & base["cc"]].reset_index(drop=True)
    rows = []
    for kw, lab in ((dict(), "S0"), (dict(hba1c_conv=True, bp_conv=True), "S1")):
        dd = scenario(naive, **kw)
        m = fit_lr(dd[dd["year"].isin(REF_YEARS)])
        for y in EVAL_YEARS:
            t = dd[dd["year"] == y]
            p = pred_lr(m, t)
            rows.append(dict(year=y, scenario=lab, n=len(t), prev=t["y"].mean(), CITL=citl(t["y"].values, p),
                             ICI=ici(t["y"].values, p)))
    nv = pd.DataFrame(rows)
    nv.to_csv(RES / "naive_cohort.csv", index=False)
    print(nv.round(3).to_string())


if __name__ == "__main__":
    main()
