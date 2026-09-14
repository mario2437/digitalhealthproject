"""05. 부차 결과: 대사증후군(NCEP-ATP III, 한국인 허리둘레 90/85 cm)으로 측정 변경 전파 확인.

S0 원자료 / S1 조화: 중성지방 2016-2021 값에 전환식(새 분석법 척도), 2021+ 혈압 전환식(예측변수·혈압 기준 모두)
HDL(2022 장비·시약 변경)은 전환식이 없어 기술만 한다.
산출: results/mets_yearly.csv, mets_components.csv
"""
import numpy as np
import pandas as pd

from common import (ALL_YEARS, EVAL_YEARS, REF_YEARS, SEED, boot_indices, citl, ici, out_dirs,
                    pct_ci, scenario, tg_harmonized)
from models import fit_lr, pred_lr
from sklearn.metrics import roc_auc_score

RES, _ = out_dirs()


def mets_frame(d, harmonize):
    o = scenario(d, bp_conv=harmonize)
    tg = tg_harmonized(o) if harmonize else o["HE_TG"]
    m = o["male"] == 1
    c_wc = ((o["wc"] >= 90) & m) | ((o["wc"] >= 85) & ~m)
    c_tg = tg >= 150
    c_hdl = ((o["HE_HDL_st2"] < 40) & m) | ((o["HE_HDL_st2"] < 50) & ~m)
    c_bp = (o["sbp"] >= 130) | (o["dbp"] >= 85)
    c_glu = o["HE_glu"] >= 100
    o["y"] = ((c_wc.astype(int) + c_tg + c_hdl + c_bp + c_glu) >= 3).astype(int)
    o["c_tg"], o["c_hdl"], o["c_bp"], o["c_wc"], o["c_glu"] = c_tg, c_hdl, c_bp, c_wc, c_glu
    return o


def main():
    base = pd.read_pickle(RES / "base_cohort.pkl")
    base = base[~base["undx_harm"] & base["cc"]].dropna(subset=["HE_TG", "HE_HDL_st2"]).reset_index(drop=True)
    rows, comp = [], []
    for harm, lab in ((False, "S0"), (True, "S1")):
        d = mets_frame(base, harm)
        m = fit_lr(d[d["year"].isin(REF_YEARS)])
        for y in ALL_YEARS:
            t = d[d["year"] == y]
            comp.append(dict(scenario=lab, year=y, n=len(t), mets=t["y"].mean(),
                             **{c: t[c].mean() for c in ["c_wc", "c_tg", "c_hdl", "c_bp", "c_glu"]}))
            if y not in EVAL_YEARS:
                continue
            yy, p = t["y"].values, pred_lr(m, t)
            bs = [citl(yy[i], p[i]) for i in boot_indices(len(yy), seed=SEED + y)]
            lo, hi = pct_ci(bs)
            rows.append(dict(scenario=lab, year=y, n=len(t), prev=yy.mean(), mean_pred=p.mean(),
                             OE=yy.mean() / p.mean(), AUROC=roc_auc_score(yy, p), CITL=citl(yy, p),
                             CITL_lo=lo, CITL_hi=hi, ICI=ici(yy, p)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "mets_yearly.csv", index=False)
    cp = pd.DataFrame(comp)
    cp.to_csv(RES / "mets_components.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    print(cp.round(3).to_string())


if __name__ == "__main__":
    main()
