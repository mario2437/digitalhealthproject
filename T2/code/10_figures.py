"""10단계: 그림 (2개)과 원고용 표.

그림 1  방법별 4범주 kappa와 초가공 이진 F1 (95% 부트스트랩 CI, 점선 = SAP 성공 기준)
그림 2  방법별 가중 평균 UPF% 추정 오차 (정답 매핑 대비 %p, 음영 = ±3%p 기준)
표      RESULTS_DIR/table_metrics.md, table_hard_cases.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import plotstyle  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

LABEL = {"RULE": "규칙", "ML-LR": "ML 로지스틱", "ML-SVM": "ML 선형SVM", "ML-GBM": "ML LightGBM",
         "LLM-0@Qwen2.5-7B-Instruct": "LLM 7B 제로샷", "LLM-FS@Qwen2.5-7B-Instruct": "LLM 7B 퓨샷",
         "LLM-0@Qwen2.5-14B-Instruct": "LLM 14B 제로샷(탐색)", "LLM-FS@Qwen2.5-14B-Instruct": "LLM 14B 퓨샷(탐색)",
         "RULE_REVIEW": "규칙+1인 검토(탐색)"}


def main():
    C.ensure_dirs()
    met = pd.read_csv(C.RESULTS_DIR / "metrics.csv", encoding="utf-8-sig")
    met = met[met.method.isin(LABEL)].reset_index(drop=True)
    met["label"] = met.method.map(LABEL)
    ypos = np.arange(len(met))[::-1]
    col = [plotstyle.PALETTE[4] if e else plotstyle.PALETTE[0] for e in met.exploratory]

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.COL_W, 0.24 * len(met) + 0.7), sharey=True)
    for ax, key, thr, title in ((axes[0], "kappa4", 0.70, "4범주 kappa"), (axes[1], "f1_upf", 0.85, "초가공 여부 F1")):
        for i, r in met.iterrows():
            ax.plot([r[f"{key}_lo"], r[f"{key}_hi"]], [ypos[i]] * 2, color=col[i], lw=1.2)
            ax.plot(r[key], ypos[i], "o", ms=4, color=col[i], mfc=col[i] if not r.exploratory else "white")
        ax.axvline(thr, color="#888888", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlim(0.5, 1)
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    axes[0].set_yticks(ypos, met.label)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig1_classifier_comparison.png")
    plt.close(fig)

    imp = pd.read_csv(C.RESULTS_DIR / "upf_impact.csv", encoding="utf-8-sig")
    imp = imp[imp.method.isin(LABEL)].set_index("method").loc[[m for m in met.method if m in set(imp.method)]]
    fig, ax = plt.subplots(figsize=(plotstyle.COL_W, 0.24 * len(imp) + 0.8))
    yp = np.arange(len(imp))[::-1]
    ax.axvspan(-3, 3, color="#eeeeee", lw=0)
    ax.axvline(0, color="#888888", lw=0.8)
    for j, (g, mk) in enumerate((("전체", "o"), ("19-39세", "s"), ("40-59세", "^"))):
        ax.plot(imp[f"diff_{g}"], yp + (0.2 - 0.2 * j), mk, ms=4, ls="none", color=plotstyle.PALETTE[j], label=g)
    ax.set_yticks(yp, [LABEL[m] for m in imp.index])
    ax.set_xlabel("가중 평균 UPF% 오차 (방법 - 정답 매핑, %p)")
    ax.legend(loc="lower right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "fig2_upf_error.png")
    plt.close(fig)

    tb = ["| 방법 | kappa (95% CI) | macro-F1 | 초가공 F1 (95% CI) | 에너지가중 정확도 |", "|---|---|---|---|---|"]
    for _, r in met.iterrows():
        tb.append(f"| {r.label} | {r.kappa4:.2f} ({r.kappa4_lo:.2f}-{r.kappa4_hi:.2f}) | {r.macro_f1:.2f} | "
                  f"{r.f1_upf:.2f} ({r.f1_upf_lo:.2f}-{r.f1_upf_hi:.2f}) | {r.energy_w_acc:.2f} |")
    (C.RESULTS_DIR / "table_metrics.md").write_text("\n".join(tb) + "\n")
    hc = pd.read_csv(C.RESULTS_DIR / "hard_cases.csv", dtype={"N_FCODE": str}, encoding="utf-8-sig")
    cols = [m for m in LABEL if m in hc.columns]
    tb = ["| 사례 | 정답 | " + " | ".join(LABEL[c] for c in cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for _, r in hc.iterrows():
        tb.append(f"| {r['사례']} | {r.nova}{'?' if r.uncertain else ''} | " + " | ".join(str(r[c]) for c in cols) + " |")
    (C.RESULTS_DIR / "table_hard_cases.md").write_text("\n".join(tb) + "\n")
    print("그림·표 저장:", C.FIGURES_DIR)


if __name__ == "__main__":
    main()
