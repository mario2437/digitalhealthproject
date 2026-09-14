"""08. 논문 그림 (최대 2개).

fig1_artifact_signal.png : 결과별 ΔAUROC(T+A − T) [95% CI], 무작위 CV vs 패널 층화 + A 단독 AUROC 표기
fig2_mitigation.png      : 흔적의 겉보기 이득 Δ_A = AUROC(B+A) − AUROC(B), 평가 방식 M0~M3 비교
색 + 마커 모양을 함께 써서 색각 이상/흑백 인쇄에서도 구분되게 함.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import plotstyle  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from t31_common import FIGURES, OUTCOMES, RESULTS  # noqa: E402

order = list(OUTCOMES.values())
ypos = np.arange(len(order))[::-1]
INK, MUTED = "#333333", "#777777"

# ---- 그림 1
d = pd.read_csv(RESULTS / "h1_delta_primary.tsv", sep="\t")
d = d[d.comparison == "T+A - T"]
perf = pd.read_csv(RESULTS / "h1_perf_primary.tsv", sep="\t")
aonly = perf[perf.model == "A"].set_index("결과").auroc
fig, ax = plt.subplots(figsize=(plotstyle.COL_W, 2.2))
for k, (strat, lab, col, mk) in enumerate([(False, "무작위 5-겹 CV", "#9aa3ab", "o"),
                                           (True, "패널 층화 AUROC", plotstyle.PALETTE[1], "s")]):
    s = d[d.panel_stratified == strat].set_index("결과").loc[order]
    yy = ypos + (0.14 if k == 0 else -0.14)
    ax.errorbar(s.delta, yy, xerr=[s.delta - s.ci_lo, s.ci_hi - s.delta], fmt=mk, color=col, ms=4.5,
                lw=1.2, capsize=0, label=lab)
ax.axvline(0, color=MUTED, lw=0.8)
ax.axvline(0.02, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(ypos, [f"{o}  (A 단독 {aonly[o]:.2f})" for o in order])
ax.set_xlabel(r"$\Delta$AUROC (원발암+흔적 - 원발암)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
ax.grid(axis="x", color="#e5e5e5", lw=0.6)
ax.set_axisbelow(True)
fig.savefig(FIGURES / "fig1_artifact_signal.png")

# ---- 그림 2
m = pd.read_csv(RESULTS / "m_mitigation.tsv", sep="\t")
m = m[m.contrast == "Δ_A"]
strats = [("M0 무작위 CV", "M0 무작위 CV", "#9aa3ab", "o"), ("M1 패널 층화", "M1 패널 층화", plotstyle.PALETTE[1], "s"),
          ("M2 IMPACT468 제한", "M2 IMPACT468만", plotstyle.PALETTE[2], "^"),
          ("M3 시간 분할 시험", "M3 시간 분할 시험", plotstyle.PALETTE[0], "D")]
fig, ax = plt.subplots(figsize=(plotstyle.COL_W, 2.4))
offs = np.linspace(0.27, -0.27, len(strats))
for (key, lab, col, mk), off in zip(strats, offs):
    s = m[m.strategy == key].set_index("결과").loc[order]
    ax.errorbar(s.delta, ypos + off, xerr=[s.delta - s.ci_lo, s.ci_hi - s.delta], fmt=mk, color=col, ms=4,
                lw=1.0, capsize=0, label=lab)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(ypos, order)
ax.set_xlabel(r"흔적 변수의 이득 $\Delta$AUROC (B+A - B)")
ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
ax.grid(axis="x", color="#e5e5e5", lw=0.6)
ax.set_axisbelow(True)
fig.savefig(FIGURES / "fig2_mitigation.png")
print("saved", FIGURES)
