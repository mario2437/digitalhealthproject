"""D5 (사후 요약) + 그림.

사후 요약(SAP_addendum에 없던 집계, '사후 탐색'으로 표기):
  - 4컷×5장기(20점)에서 전략별 외부 ΔAUROC(Sk − S1)의 평균·최소·≥0 개수
  - 내부 부풀림 = 내부 ΔAUROC(E0, Sk − S1) − 외부 ΔAUROC(Sk − S1) 평균
그림:
  fig_d1_strategies.png : 전략별 내부(E0) 이득 vs 외부(시험) 이득 (S1 대비), 대각선=정직
  fig_d3_importance.png : 묶음 퍼뮤테이션 ΔAUROC, 교차검증 vs 시간 분할 시험 (5장기 평균)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
import plotstyle  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from deep_common import FIGURES, ORGAN_KEYS, RESULTS  # noqa: E402

m = pd.read_csv(RESULTS / "d1_metrics.tsv", sep="\t")
m = m[m.outcome.isin(ORGAN_KEYS)]
LAB = {"S0": "S0 흔적 포함", "S1": "S1 흔적 제거", "S1b": "S1b TMB도 제외", "S2": "S2 추적≥24개월 학습",
       "S2b": "S2b 추적≥36개월 학습", "S3": "S3 패널 균형 가중", "S4": "S4 시기 잔차화", "S5a": "S5a 흔적 고정값",
       "S5b": "S5b 흔적 주변화", "S6": "S6 추적 주변화", "S7": "S7 흔적·추적 주변화"}
rows = []
base = m[m.strategy == "S1"].set_index(["cut", "outcome"])
for s, g in m.groupby("strategy"):
    g = g.set_index(["cut", "outcome"])
    ext = g.auc_test - base.auc_test
    inn = g.auc_E0 - base.auc_E0
    rows.append(dict(strategy=s, ext_mean=ext.mean(), ext_min=ext.min(), ext_n_ge0=int((ext >= 0).sum()),
                     int_mean=inn.mean(), inflation=(inn - ext).mean(),
                     signed_gap=(g.auc_E0 - g.auc_test).mean(),
                     ext_mean_c70=ext.xs("c70").mean(), int_mean_c70=inn.xs("c70").mean(),
                     oe_c70=g.oe_test.xs("c70").mean(), oe_complete_c70=g.oe_test_complete.xs("c70").mean()))
ph = pd.DataFrame(rows).set_index("strategy").loc[list(LAB)]
ph.to_csv(RESULTS / "d5_posthoc_strategy_summary.tsv", sep="\t", float_format="%.4f")
print(ph.round(4).to_string())

# ---- 그림 A: 내부 vs 외부 이득
show = ["S0", "S1", "S2", "S4", "S5a", "S5b", "S7"]
MK = {"S0": ("#a8511f", "o"), "S1": ("#5c6873", "s"), "S2": ("#7a3b62", "v"), "S4": ("#0e6a63", "^"),
      "S5a": ("#0e6a63", "D"), "S5b": ("#1f5e8c", "D"), "S7": ("#1f5e8c", "P")}
fig, ax = plt.subplots(figsize=(plotstyle.COL_W, 2.6))
lim = (-0.02, 0.026)
ax.plot(lim, lim, color="#999999", lw=0.8, ls="--", zorder=0)
ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
offs = {"S0": (4, -9), "S1": (6, -11), "S2": (-8, -11), "S4": (-4, -11), "S5a": (-10, 5), "S5b": (5, 4), "S7": (-13, 5)}
for s in show:
    g = m[m.strategy == s].set_index(["cut", "outcome"])
    inn = (g.auc_E0 - base.auc_E0).groupby(level=0).mean()
    ext = (g.auc_test - base.auc_test).groupby(level=0).mean()
    col, mk = MK[s]
    ax.scatter(inn, ext, s=9, color=col, marker=mk, alpha=0.35, lw=0)
    ax.scatter(inn.mean(), ext.mean(), s=38, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3)
    ax.annotate(s, (inn.mean(), ext.mean()), xytext=offs[s], textcoords="offset points", fontsize=6.5,
                color="#333333")
ax.set_xlim(lim)
ax.set_ylim(-0.012, 0.016)
ax.set_xlabel("내부(무작위 CV) AUROC 이득, S1 대비")
ax.set_ylabel("시간 분할 시험 AUROC 이득, S1 대비")
ax.text(0.024, 0.0135, "내부=외부", fontsize=6, color="#777777", ha="right")
ax.grid(color="#eeeeee", lw=0.5)
ax.set_axisbelow(True)
fig.savefig(FIGURES / "fig_d1_strategies.png")

# ---- 그림 B: 퍼뮤테이션 중요도 내부 vs 외부
cv = pd.read_csv(RESULTS / "d3_perm_cv.tsv", sep="\t")
ex = pd.read_csv(RESULTS / "d3_perm_external.tsv", sep="\t")
items = [("A_시기", "흔적: 패널·등록 순번"), ("A_품질", "흔적: 검체 품질"), ("T", "원발암 종류"), ("G", "유전체 변이"),
         ("C", "임상(나이·TMB 등)"), ("NULL_RANKINT", "영 특징(무작위 정수)")]
fig, ax = plt.subplots(figsize=(plotstyle.COL_W, 2.0))
yy = np.arange(len(items))[::-1]
summ = {}
for k, (df, lab, col, mk, off) in enumerate([(cv, "5-겹 교차검증", "#a8511f", "o", 0.14),
                                             (ex, "시간 분할 시험", "#1f5e8c", "D", -0.14)]):
    d = df[df.outcome.isin(ORGAN_KEYS)].groupby("item")[["delta_auc", "ci_lo", "ci_hi"]].mean()
    d = d.loc[[i for i, _ in items]]
    summ[lab] = d.delta_auc.round(4).to_dict()
    ax.errorbar(d.delta_auc, yy + off, xerr=[d.delta_auc - d.ci_lo, d.ci_hi - d.delta_auc], fmt=mk, color=col,
                ms=3.8, lw=1.0, capsize=0, label=lab)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_yticks(yy, [l for _, l in items])
ax.set_xlabel(r"퍼뮤테이션 $\Delta$AUROC (5개 장기 평균)")
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
ax.grid(axis="x", color="#eeeeee", lw=0.5)
ax.set_axisbelow(True)
fig.savefig(FIGURES / "fig_d3_importance.png")
(RESULTS / "d5_posthoc_summary.json").write_text(json.dumps({"퍼뮤테이션 5장기 평균": summ}, ensure_ascii=False,
                                                            indent=2))
print(summ)
