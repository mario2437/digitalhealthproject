"""9단계: 분류 오류가 개인별 UPF% 모집단 추정에 주는 영향 (SAP 6절) + 성공 기준 판정 (SAP 7절).

매핑: 정답셋 코드에는 각 방법의 예측, 정답셋 밖 코드에는 모든 방법 공통으로 RULE 라벨.
기준: 정답셋 코드에 정답 라벨 + 나머지 RULE 라벨.
대상: 19-59세, wt_tot>0, 총 에너지 500-5,000 kcal. 가중치 wt_tot/4(2019-2022 통합).

출력  RESULTS_DIR/upf_impact.csv, criteria.json
"""
import json

import numpy as np
import pandas as pd

import config as C

CORE = ["RULE", "ML-LR", "ML-SVM", "ML-GBM", "LLM-0@Qwen2.5-7B-Instruct", "LLM-FS@Qwen2.5-7B-Instruct"]


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def main():
    rc = pd.read_pickle(C.CACHE_DIR / "rc.pkl")[["ID", "N_FCODE", "NF_EN"]]
    al = pd.read_pickle(C.CACHE_DIR / "all.pkl")[["ID", "age", "wt_tot"]]
    gold = pd.read_csv(C.GOLD, dtype={"N_FCODE": str}, encoding="utf-8-sig")
    rule = pd.read_csv(C.PRED_DIR / "pred_RULE.csv", dtype={"N_FCODE": str}, encoding="utf-8-sig")
    base = rule.set_index("N_FCODE").pred

    person = rc.groupby("ID").NF_EN.sum().rename("en").reset_index().merge(al, on="ID")
    person = person[(person.wt_tot > 0) & person.en.between(C.ENERGY_MIN, C.ENERGY_MAX)].copy()
    person["w"] = person.wt_tot / len(C.YEARS)
    person["young"] = person.age <= C.YOUNG_MAX
    rc = rc[rc.ID.isin(person.ID)]
    en_by_code = rc.groupby("N_FCODE").NF_EN.sum()
    gold_share = float(en_by_code.reindex(gold.N_FCODE).fillna(0).sum() / en_by_code.sum() * 100)
    # 사람×코드 에너지 행렬(희소 대신 groupby)
    pc = rc.groupby(["ID", "N_FCODE"]).NF_EN.sum().reset_index()

    def upf_pct(mapping):
        is4 = pc.N_FCODE.map(mapping).fillna(1).eq(4)
        e4 = pc.NF_EN.where(is4, 0).groupby(pc.ID).sum()
        return 100 * person.ID.map(e4).fillna(0).to_numpy() / person.en.to_numpy()

    def mapping_with(codes, labels):
        m = base.copy()
        m.loc[codes] = labels
        return m

    ref = upf_pct(mapping_with(gold.N_FCODE.to_numpy(), gold.nova.astype(int).to_numpy()))
    rows = []
    methods = {"GOLD(기준)": None}
    for f in sorted(C.PRED_DIR.glob("pred_*.csv")):
        if not f.stem.endswith("_all"):
            methods[f.stem[5:]] = f
    for name, f in methods.items():
        if f is None:
            u = ref
        else:
            p = pd.read_csv(f, dtype={"N_FCODE": str}, encoding="utf-8-sig")
            p = p[p.N_FCODE.isin(gold.N_FCODE)]
            u = upf_pct(mapping_with(p.N_FCODE.to_numpy(), p.pred.astype(int).to_numpy()))
        r = {"method": name}
        for g, m in (("전체", np.ones(len(person), bool)), ("19-39세", person.young.to_numpy()),
                     ("40-59세", ~person.young.to_numpy())):
            r[f"UPF_{g}"] = wmean(u[m], person.w.to_numpy()[m])
            r[f"diff_{g}"] = r[f"UPF_{g}"] - wmean(ref[m], person.w.to_numpy()[m])
        r["person_MAE"] = wmean(np.abs(u - ref), person.w.to_numpy())
        r["person_corr"] = float(np.corrcoef(u, ref)[0, 1])
        rows.append(r)
    imp = pd.DataFrame(rows)
    imp.round(3).to_csv(C.RESULTS_DIR / "upf_impact.csv", index=False, encoding="utf-8-sig")
    print(f"대상 {len(person):,}명, 정답셋 코드의 에너지 비율 {gold_share:.1f}%")
    print(imp.round(2).to_string(index=False))

    met = pd.read_csv(C.RESULTS_DIR / "metrics.csv", encoding="utf-8-sig")
    core = met[met.method.isin(CORE)].sort_values("kappa4", ascending=False)
    best = core.iloc[0]
    bi = imp.set_index("method").loc[best.method]
    crit = {
        "n_persons": int(len(person)), "gold_energy_share_pct": round(gold_share, 1),
        "best_method": best.method,
        "S1_kappa4": [round(best.kappa4, 3), [round(best.kappa4_lo, 3), round(best.kappa4_hi, 3)],
                      bool(best.kappa4 >= 0.70)],
        "S2_f1_upf": [round(best.f1_upf, 3), bool(best.f1_upf >= 0.85)],
        "S3_upf_diff_pp": [{g: round(float(bi[f"diff_{g}"]), 2) for g in ("전체", "19-39세", "40-59세")},
                           bool(max(abs(bi[f"diff_{g}"]) for g in ("전체", "19-39세", "40-59세")) <= 3)],
    }
    crit["all_met"] = crit["S1_kappa4"][2] and crit["S2_f1_upf"][1] and crit["S3_upf_diff_pp"][1]
    (C.RESULTS_DIR / "criteria.json").write_text(json.dumps(crit, ensure_ascii=False, indent=2))
    print(json.dumps(crit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
