"""07. 사전 지정 성공 기준(SAP 7절) 판정. 임계값은 SAP와 동일하게 코드에 고정.

산출: results/criteria.csv
"""
import pandas as pd

from common import out_dirs

RES, _ = out_dirs()


def main():
    ym = pd.read_csv(RES / "yearly_metrics.csv")
    ref = pd.read_csv(RES / "reference_cv.csv")
    dom = pd.read_csv(RES / "domain_auc.csv")
    sh = pd.read_csv(RES / "decomposition_shares.csv").set_index("year")
    dec = pd.read_csv(RES / "decomposition.csv")
    out = []

    def add(cid, desc, met, detail):
        out.append(dict(criterion=cid, description=desc, result=met, detail=detail))

    # C0
    e0 = sh.loc[[2019, 2020, 2021], "E0"]
    add("C0", "2019-2021 각 연도 E_0 < 0.02", "충족" if (e0 < 0.02).all() else "미충족",
        "; ".join(f"{y}: {v:.3f}" for y, v in e0.items()))
    # C1
    s22 = sh.loc[2022, "sh_meas"]
    add("C1", "2022 측정 몫 >= 50%", "충족" if s22 >= 0.5 else "미충족",
        f"{s22:.3f} (95% CI {sh.loc[2022, 'sh_meas_lo']:.3f}-{sh.loc[2022, 'sh_meas_hi']:.3f})")
    # C2
    s = sh.loc[[2023, 2024], "sh_meas"]
    nmet = int((s >= 0.5).sum())
    add("C2", "2023·2024 측정 몫 >= 50%", {2: "충족", 1: "부분 충족", 0: "미충족"}[nmet],
        "; ".join(f"{y}: {v:.3f} ({sh.loc[y, 'sh_meas_lo']:.3f}-{sh.loc[y, 'sh_meas_hi']:.3f})"
                  for y, v in s.items()))
    # C3
    g = lambda o, sc, y: ym[(ym.outcome == o) & (ym.scenario == sc) & (ym.year == y)].iloc[0]  # noqa: E731
    parts, ok = [], True
    for y in (2022, 2023, 2024):
        a0, a1, f0 = abs(g("hba1c", "S0", y).CITL), abs(g("hba1c", "S1", y).CITL), abs(g("fpg", "S0", y).CITL)
        red = 1 - a1 / a0
        ok_y = (a0 >= 0.30) and (f0 < 0.15) and (red >= 0.5)
        ok &= ok_y
        parts.append(f"{y}: |HbA1c CITL| {a0:.3f}->{a1:.3f} (감소 {red:.0%}), |FPG CITL| {f0:.3f} -> {'O' if ok_y else 'X'}")
    add("C3", "HbA1c 단독 |CITL|>=0.30, FPG 단독 |CITL|<0.15, 전환식으로 HbA1c |CITL| 50% 이상 감소",
        "충족" if ok else "미충족", "; ".join(parts))
    # C4
    da = dom[dom.bp_conv == False].set_index("year")["domain_auc_lr"]  # noqa: E712
    cov = sh["sh_cov"].dropna()
    ok4a, ok4b = (da < 0.60).all(), (cov < 0.25).all()
    add("C4", "모든 연도 도메인 AUROC<0.60 그리고 공변량 몫<25%",
        "충족" if ok4a and ok4b else ("부분(공변량 몫만 충족)" if ok4b else "미충족"),
        "도메인 AUROC " + ", ".join(f"{y}:{v:.3f}" for y, v in da.items()) +
        " | 공변량 몫 " + ", ".join(f"{y}:{v:.3f}" for y, v in cov.items()))
    # C5
    parts, ok = [], True
    for y in (2021, 2022, 2023, 2024):
        dlt = g("composite", "S1b", y).ICI - g("composite", "S0", y).ICI
        ok &= abs(dlt) < 0.01
        parts.append(f"{y}: {dlt:+.4f}")
    add("C5", "혈압 전환식 단독 |ΔICI|<0.01 (2021-2024)", "충족" if ok else "미충족", "; ".join(parts))
    # C6
    rauc = ref[ref.outcome == "composite"].AUROC.iloc[0]
    parts, ok = [], True
    for y in range(2019, 2025):
        r = g("composite", "S0", y)
        ok_y = abs(r.AUROC - rauc) <= 0.03 and abs(r.AUROC - r.mbc) <= 0.03
        ok &= ok_y
        parts.append(f"{y}: AUROC {r.AUROC:.3f}, mbc {r.mbc:.3f}")
    add("C6", f"AUROC가 기준 CV({rauc:.3f})±0.03 이내, |AUROC-mbc|<=0.03", "충족" if ok else "미충족",
        "; ".join(parts))
    # 코로나19 표시
    c = dec[dec.scenario == "S1"].set_index("year")["CITL"]
    lo, hi = c.loc[[2019, 2022, 2023, 2024]].min(), c.loc[[2019, 2022, 2023, 2024]].max()
    flags = [y for y in (2020, 2021) if c[y] < lo - 0.10 or c[y] > hi + 0.10]
    add("COVID", "S1 CITL이 인접 연도 범위를 0.10 이상 벗어남(기술적 표시)",
        "표시됨: " + ",".join(map(str, flags)) if flags else "표시 없음",
        f"범위 {lo:.3f}~{hi:.3f}; 2020 {c[2020]:.3f}, 2021 {c[2021]:.3f}")
    df = pd.DataFrame(out)
    df.to_csv(RES / "criteria.csv", index=False)
    pd.set_option("display.max_colwidth", 300)
    print(df.to_string())


if __name__ == "__main__":
    main()
