"""8단계: 정답셋 대비 분류 성능 (SAP 5절).

주 지표: 4범주 Cohen's kappa. 핵심 부 지표: 초가공(4군) 이진 F1.
기타: 정확도, macro-F1, 이진 kappa, 에너지가중 정확도, 층별 kappa, uncertain=0 코드 kappa, 혼동행렬.
95% CI: 코드 단위 부트스트랩(모든 방법이 같은 재표본을 공유 → RULE 대비 쌍체 차이).

입력  GOLD, PRED_DIR/pred_*.csv (gold 범위)
출력  RESULTS_DIR/metrics.csv, metrics_boot_diff.csv, confusion_<방법>.csv, hard_cases.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

import config as C

ORDER = ["RULE", "ML-LR", "ML-SVM", "ML-GBM", "LLM-0@Qwen2.5-7B-Instruct", "LLM-FS@Qwen2.5-7B-Instruct",
         "LLM-0@Qwen2.5-14B-Instruct", "LLM-FS@Qwen2.5-14B-Instruct", "RULE_REVIEW"]
EXPLORATORY = {"LLM-0@Qwen2.5-14B-Instruct", "LLM-FS@Qwen2.5-14B-Instruct", "RULE_REVIEW"}
HARD = {"29116": "햇반(즉석밥)", "01328": "즉석 잡곡밥", "04023": "두유", "95503": "소주(참이슬)",
        "15029": "막걸리", "16010": "고추장(개량)", "09060": "후라이드치킨", "09356": "오리고기(훈제 응답)",
        "13030": "플레인 호상 요구르트", "01204": "누룽지", "17013": "고기만두", "01084": "식빵"}


def kappa(y, p, labels):
    k = cohen_kappa_score(y, p, labels=labels)
    return 0.0 if np.isnan(k) else k


def core(y, p, w):
    return {
        "kappa4": kappa(y, p, [1, 2, 3, 4]),
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=[1, 2, 3, 4], average="macro", zero_division=0),
        "f1_upf": f1_score(y == 4, p == 4, zero_division=0),
        "kappa_upf": kappa((y == 4).astype(int), (p == 4).astype(int), [0, 1]),
        "energy_w_acc": float((w * (y == p)).sum() / w.sum()),
    }


def load_preds(gold):
    preds = {}
    for f in sorted(C.PRED_DIR.glob("pred_*.csv")):
        name = f.stem[5:]
        if name.endswith("_all"):
            continue
        p = pd.read_csv(f, dtype={"N_FCODE": str}, encoding="utf-8-sig")
        m = gold[["N_FCODE"]].merge(p[["N_FCODE", "pred"]], on="N_FCODE", how="left")
        assert m.pred.notna().all(), f"{name}: 정답셋 코드 중 예측 없음"
        preds[name] = m.pred.astype(int).to_numpy()
    return {k: preds[k] for k in ORDER if k in preds} | {k: v for k, v in preds.items() if k not in ORDER}


def main():
    gold = pd.read_csv(C.GOLD, dtype={"N_FCODE": str}, encoding="utf-8-sig")
    y = gold.nova.astype(int).to_numpy()
    w = gold["에너지비율_pct"].to_numpy()
    preds = load_preds(gold)
    rng = np.random.default_rng(C.SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(C.N_BOOT)]

    rows, diffs, boot_store = [], [], {}
    for name, p in preds.items():
        r = {"method": name, "exploratory": int(name in EXPLORATORY), "n": len(y)} | core(y, p, w)
        for s in ("top", "tail"):
            m = (gold.stratum == s).to_numpy()
            r[f"kappa4_{s}"] = kappa(y[m], p[m], [1, 2, 3, 4])
            r[f"f1_upf_{s}"] = f1_score(y[m] == 4, p[m] == 4, zero_division=0)
        m = (gold.uncertain == 0).to_numpy()
        r["kappa4_certain"], r["n_certain"] = kappa(y[m], p[m], [1, 2, 3, 4]), int(m.sum())
        bs = np.array([[v for v in core(y[i], p[i], w[i]).values()] for i in idx])
        boot_store[name] = bs
        for j, key in enumerate(core(y, p, w).keys()):
            r[f"{key}_lo"], r[f"{key}_hi"] = np.percentile(bs[:, j], [2.5, 97.5])
        rows.append(r)
        cm = pd.DataFrame(confusion_matrix(y, p, labels=[1, 2, 3, 4]),
                          index=[f"정답{k}" for k in range(1, 5)], columns=[f"예측{k}" for k in range(1, 5)])
        cm.to_csv(C.RESULTS_DIR / f"confusion_{name}.csv", encoding="utf-8-sig")
    met = pd.DataFrame(rows)
    met.round(4).to_csv(C.RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    if "RULE" in boot_store:
        base = boot_store["RULE"]
        base_pt = met.set_index("method").loc["RULE"]
        for name, bs in boot_store.items():
            if name == "RULE":
                continue
            pt = met.set_index("method").loc[name]
            for j, key in enumerate(["kappa4", "acc", "macro_f1", "f1_upf"]):
                jj = list(core(y, y, w).keys()).index(key)
                d = bs[:, jj] - base[:, jj]
                diffs.append({"method": name, "metric": key, "diff_vs_RULE": pt[key] - base_pt[key],
                              "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
        pd.DataFrame(diffs).round(4).to_csv(C.RESULTS_DIR / "metrics_boot_diff.csv", index=False,
                                            encoding="utf-8-sig")

    hc = gold[gold.N_FCODE.isin(HARD)][["N_FCODE", "식품명", "nova", "uncertain"]].copy()
    hc.insert(1, "사례", hc.N_FCODE.map(HARD))
    for name, p in preds.items():
        hc[name] = p[gold.N_FCODE.isin(HARD).to_numpy()]
    hc.to_csv(C.RESULTS_DIR / "hard_cases.csv", index=False, encoding="utf-8-sig")
    show = ["method", "kappa4", "kappa4_lo", "kappa4_hi", "macro_f1", "f1_upf", "f1_upf_lo", "f1_upf_hi",
            "acc", "energy_w_acc", "kappa4_top", "kappa4_tail", "kappa4_certain"]
    print(met[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
