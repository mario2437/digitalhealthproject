"""8단계: 논문 빈칸에 들어갈 핵심 수치를 한 파일로 모은다 → results/paper_numbers.txt, table1_top_edges.csv"""
import pandas as pd

from common import RESULTS_DIR

R = RESULTS_DIR
lines = []
coh = pd.read_csv(R / "cohort.csv")
lines.append(f"네트워크 코호트 N={len(coh)}, 원발 {int((coh.sample_type=='Primary').sum())}, "
             f"조직형 {coh.histology.value_counts().to_dict()}, 부담 중앙값 {coh.burden.median()} (IQR {coh.burden.quantile(.25)}–{coh.burden.quantile(.75)})")
lines.append(f"부위 불명 라벨 {100*coh.site_UNSPECIFIED.mean():.1f}%")
lines.append("\n[주변 vs 보정 요약]\n" + pd.read_csv(R / "marginal_vs_adjusted_summary.csv").T.to_string())
lines.append("\n[성공 기준]\n" + pd.concat([pd.read_csv(R / "criteria_C1_C2.csv"), pd.read_csv(R / "criteria_C3.csv")]).to_string(index=False))
pm = pd.read_csv(R / "pairs_all_methods.csv")
t1 = pm[pm.adj_sig].copy()
t1["abs"] = t1.adj_logor.abs()
t1 = pd.concat([t1[t1.adj_logor > 0].sort_values("adj_logor", ascending=False).head(6),
                t1[t1.adj_logor < 0].sort_values("adj_logor")])
t1 = t1[["label", "n_ij", "lift", "marg_or", "adj_or", "adj_lo", "adj_hi", "adj_fdr", "z", "cb_fdr", "weight"]]
t1.to_csv(R / "table1_top_edges.csv", index=False)
lines.append("\n[표 1 후보]\n" + t1.round(3).to_string(index=False))
lines.append("\n[연관 규칙]\n" + pd.read_csv(R / "rules_top.csv").round(3).to_string(index=False))
lines.append("\n[모듈]\n" + pd.read_csv(R / "modules.csv").to_string(index=False) + "\n" + pd.read_csv(R / "modules_summary.csv").round(3).to_string(index=False))
lines.append("\n[유전형 전역]\n" + pd.read_csv(R / "geno_global.csv").round(3).T.to_string())
for c in ["main_EGFR_vs_KRAS", "S4_allsamples_EGFR_vs_KRAS", "S5_canonical_EGFR_vs_KRAS", "aux_STK11KEAP1_vs_neither"]:
    lines.append(f"\n[{c}]\n" + pd.read_csv(R / f"geno_edges_{c}.csv").sort_values("p_perm").head(5).round(3).to_string(index=False))
lines.append("\n[전이 부담 Poisson]\n" + pd.read_csv(R / "geno_burden_poisson.csv").round(3).to_string(index=False))
gp = pd.read_csv(R / "geno_site_prev.csv")
lines.append("\n[유전형별 부위 유병률]\n" + gp.round(3).to_string(index=False))
txt = "\n".join(lines)
(R / "paper_numbers.txt").write_text(txt, encoding="utf-8")
print(txt)
