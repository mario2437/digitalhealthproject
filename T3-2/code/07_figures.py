"""7단계: 그림 1(부담 보정 동시발생 네트워크), 그림 2(EGFR vs KRAS 간선 차이).

출력: figures/fig1_network.png, figures/fig2_genotype_edges.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import plotstyle  # noqa: E402,F401
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import FIGURES_DIR, RESULTS_DIR, SITE_LABEL  # noqa: E402

MOD_COL = ["#1f5e8c", "#a8511f", "#7a3b62", "#5c6873"]

# ---------------- 그림 1 ----------------
pm = pd.read_csv(RESULTS_DIR / "pairs_main.csv")
sel = pd.read_csv(RESULTS_DIR / "site_selection.csv").query("selected").set_index("site")
mods = pd.read_csv(RESULTS_DIR / "modules.csv").set_index("site")
order = mods.sort_values(["module", "strength_pos"], ascending=[True, False]).index.tolist()
K = len(order)
ang = np.pi / 2 - 2 * np.pi * np.arange(K) / K
pos = {s: (np.cos(a), np.sin(a)) for s, a in zip(order, ang)}

fig, ax = plt.subplots(figsize=(3.4, 3.3))
lost = pm[pm.marg_sig & ~pm.adj_sig]
for _, r in lost.iterrows():
    (x1, y1), (x2, y2) = pos[r.site_i], pos[r.site_j]
    ax.plot([x1, x2], [y1, y2], color="#c9ced3", lw=0.5, ls=(0, (1, 1.5)), zorder=1)
adj = pm[pm.adj_sig].sort_values("adj_logor", key=np.abs)
for _, r in adj.iterrows():
    (x1, y1), (x2, y2) = pos[r.site_i], pos[r.site_j]
    w = 0.6 + 1.6 * min(abs(r.adj_logor), 2.4)
    if r.adj_logor > 0:
        ax.plot([x1, x2], [y1, y2], color="#2b2b2b", lw=w, alpha=0.75, solid_capstyle="round", zorder=2)
    else:
        ax.plot([x1, x2], [y1, y2], color="#1f78b4", lw=w, ls=(0, (3, 1.5)), alpha=0.9, zorder=2)
for s in order:
    x, y = pos[s]
    m = int(mods.loc[s, "module"])
    ax.scatter(x, y, s=40 + 520 * sel.loc[s, "prev"], color=MOD_COL[m - 1], edgecolor="white", lw=1.2, zorder=3)
    rr = 1.22 + (0.1 if abs(y) > 0.9 else 0)
    lx, ly = rr * x, rr * y
    ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
    ax.text(lx, ly, f"{SITE_LABEL[s]}\n{100 * sel.loc[s, 'prev']:.0f}%", ha=ha, va="center", fontsize=6.8,
            linespacing=0.95, zorder=4)
ax.set_xlim(-1.75, 1.75)
ax.set_ylim(-1.45, 1.45)
ax.set_aspect("equal")
ax.axis("off")
handles = [
    Line2D([], [], color="#2b2b2b", lw=2, label="보정 후 양의 연관"),
    Line2D([], [], color="#1f78b4", lw=2, ls=(0, (3, 1.5)), label="보정 후 음의 연관"),
    Line2D([], [], color="#b5bcc3", lw=0.8, ls=(0, (1, 1.5)), label="보정 후 사라진 연관"),
]
for m in sorted(mods["module"].unique()):
    first = SITE_LABEL[mods[mods.module == m].sort_values("strength_pos").index[-1]]
    handles.append(Line2D([], [], marker="o", ls="", color=MOD_COL[m - 1], ms=5, label=f"모듈 {m} ({first} 등)"))
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6, frameon=False,
          handlelength=1.8, columnspacing=1.0)
fig.savefig(FIGURES_DIR / "fig1_network.png")
plt.close(fig)

# ---------------- 그림 2 ----------------
ge = pd.read_csv(RESULTS_DIR / "geno_edges_main_EGFR_vs_KRAS.csv")
s4 = pd.read_csv(RESULTS_DIR / "geno_edges_S4_allsamples_EGFR_vs_KRAS.csv").set_index("label")
top = ge.sort_values("p_perm").head(8).sort_values("delta")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
y = np.arange(len(top))
ax.axvline(0, color="#8a9199", lw=0.6, zorder=0)
for k, (_, r) in enumerate(top.iterrows()):
    ax.plot([r.w_EGFR, r.w_KRAS], [k, k], color="#b5bcc3", lw=1.5, zorder=1)
ax.scatter(top.w_EGFR, y, marker="o", s=26, color="#1f5e8c", zorder=2, edgecolor="white", lw=0.6)
ax.scatter(top.w_KRAS, y, marker="s", s=22, color="#a8511f", zorder=2, edgecolor="white", lw=0.6)
labels = []
for _, r in top.iterrows():
    star = "**" if r.fdr < 0.05 else ("*" if r.fdr < 0.1 else "")
    s4f = s4.loc[r.label, "fdr"] if r.label in s4.index else np.nan
    dag = "†" if s4f < 0.1 else ""
    labels.append(f"{r.label} {star}{dag}")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("부담 보정 log OR (MH)")
gl = pd.read_csv(RESULTS_DIR / "geno_global.csv").set_index("comparison").loc["main_EGFR_vs_KRAS"]
h = [Line2D([], [], marker="o", ls="", color="#1f5e8c", ms=5, label=f"EGFR (n={int(gl.n_A)})"),
     Line2D([], [], marker="s", ls="", color="#a8511f", ms=5, label=f"KRAS (n={int(gl.n_B)})")]
ax.legend(handles=h, loc="lower right", fontsize=6.5, frameon=False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e3e6e9", lw=0.5)
ax.set_axisbelow(True)
fig.savefig(FIGURES_DIR / "fig2_genotype_edges.png")
plt.close(fig)
print("saved", FIGURES_DIR)
