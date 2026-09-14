"""06. 그림
  그림 1: 연도별 AUROC(위)와 보정 절편(아래; 복합 결과 원자료·측정조화, HbA1c 단독·FPG 단독 원자료), 문서화된 변경 표시
  그림 2: 연도별 ICI 분해(원자료 → 측정 조화 → 공변량 가중 → 절편 재보정), 회색 눈금 = 완전 보정 시 기대 ICI
산출: figures/fig1_trajectory.png, figures/fig2_decomposition.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import plotstyle  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from common import EVAL_YEARS, out_dirs  # noqa: E402

RES, FIG = out_dirs()
C = plotstyle.PALETTE
REF_X = 2017.6   # 기준(2016-2018 CV) 표시 위치
# 변경 경계선(x 위치, 라벨)
CHANGES = [(2018.5, "허리둘레 줄자\n생화학 장비"), (2020.0, "코로나19"),
           (2020.5, "진동형\n혈압계"), (2021.5, "검사기관\n(HbA1c)")]


def annotate(ax, labels=False):
    for x, lab in CHANGES:
        if x == 2020.0:
            ax.axvspan(2019.6, 2021.4, color="#d9d9d9", alpha=0.35, lw=0, zorder=0)
            if labels:
                ax.text(2019.65, 0.97, lab, transform=ax.get_xaxis_transform(), ha="left", va="top",
                        fontsize=5.5, color="#444444")
            continue
        else:
            ax.axvline(x, color="grey", ls=":", lw=0.8, zorder=0)
        if labels:
            ax.text(x, 1.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.5, color="#444444")


def fig1():
    ym = pd.read_csv(RES / "yearly_metrics.csv")
    ref = pd.read_csv(RES / "reference_cv.csv").set_index("outcome")
    yrs = np.array(EVAL_YEARS)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(plotstyle.COL_W, plotstyle.COL_W * 1.05), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.5], hspace=0.12))
    s0 = ym[(ym.outcome == "composite") & (ym.scenario == "S0")].set_index("year").loc[yrs]
    annotate(a1, labels=True)
    a1.errorbar(yrs, s0.AUROC, yerr=[s0.AUROC - s0.AUROC_lo, s0.AUROC_hi - s0.AUROC], fmt="o-",
                color=C[0], ms=3.5, lw=1.2, capsize=1.5, label="관측 AUROC")
    a1.plot(yrs, s0.mbc, "D--", mfc="white", color=C[4], ms=3, lw=0.9, label="기대 AUROC(mbc)")
    a1.plot([REF_X], [ref.loc["composite", "AUROC"]], "s", color="black", ms=3.5)
    a1.set_ylim(0.70, 0.83)
    a1.set_ylabel("AUROC")
    a1.legend(frameon=False, loc="lower right", ncol=2, fontsize=6)

    annotate(a2)
    a2.axhline(0, color="black", lw=0.7)
    series = [("composite", "S0", "전단계(원자료)", C[0], "o", "-", 1.4),
              ("composite", "S1", "전단계(공식 전환식)", C[1], "s", "-", 1.4),
              ("hba1c", "S0", "HbA1c 단독(원자료)", C[3], "^", "--", 0.9),
              ("fpg", "S0", "FPG 단독(원자료)", C[2], "v", "--", 0.9)]
    for o, sc, lab, col, mk, ls, lw in series:
        r = ym[(ym.outcome == o) & (ym.scenario == sc)].set_index("year").loc[yrs]
        if lw > 1:
            a2.fill_between(yrs, r.CITL_lo, r.CITL_hi, color=col, alpha=0.15, lw=0)
        a2.plot(yrs, r.CITL, marker=mk, ls=ls, color=col, ms=3.5, lw=lw, label=lab)
    a2.plot([REF_X], [0], "s", color="black", ms=3.5)
    a2.set_ylabel("보정 절편 (0=완전)\n(+) 과소예측 / (-) 과대예측")
    a2.set_xticks([REF_X] + list(yrs))
    a2.set_xticklabels(["기준\n16-18"] + [str(y) for y in yrs])
    a2.legend(frameon=False, loc="lower left", fontsize=5.8, ncol=2)
    a2.set_ylim(-1.15, 0.75)
    fig.savefig(FIG / "fig1_trajectory.png")
    plt.close(fig)


def fig2():
    dec = pd.read_csv(RES / "decomposition.csv")
    sh = pd.read_csv(RES / "decomposition_shares.csv").set_index("year")
    yrs = EVAL_YEARS
    scen = [("S0", "원자료", C[0]), ("S1", "+측정 조화", C[1]), ("S2", "+공변량 가중", C[2]),
            ("S3", "+절편 재보정", C[4])]
    fig, ax = plt.subplots(figsize=(plotstyle.COL_W, plotstyle.COL_W * 0.62))
    w = 0.2
    x = np.arange(len(yrs))
    for j, (sc, lab, col) in enumerate(scen):
        r = dec[dec.scenario == sc].set_index("year").loc[yrs]
        xx = x + (j - 1.5) * w
        ax.bar(xx, r.ICI, width=w * 0.9, color=col, label=lab, zorder=2)
        ax.errorbar(xx, r.ICI, yerr=[r.ICI - r.ICI_lo, r.ICI_hi - r.ICI], fmt="none", ecolor="#333333",
                    elinewidth=0.5, capsize=0, zorder=3)
        ax.scatter(xx, r.ICI_null, marker="_", s=30, color="black", lw=1.0, zorder=4,
                   label="완전 보정 기대치" if j == 0 else None)
    for i, y in enumerate(yrs):
        ax.text(i, 0.100, f"측정 {sh.loc[y, 'sh_meas']*100:.0f}%\n절편 {sh.loc[y, 'sh_prior']*100:.0f}%",
                ha="center", va="bottom", fontsize=5.5, linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yrs])
    ax.set_ylabel("ICI")
    ax.set_ylim(0, 0.12)
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.savefig(FIG / "fig2_decomposition.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("그림 저장:", FIG)
