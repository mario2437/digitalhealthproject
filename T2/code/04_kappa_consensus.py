"""4단계(학생 정답셋): 두 배정자의 일치도(Cohen's kappa)를 계산하고 합의 초안을 만든다.

입력  RESULTS_DIR/gold_rater_A.csv, gold_rater_B.csv (03단계 템플릿에 각자 nova·uncertain·rule·note를 채운 것)
출력  RESULTS_DIR/rater_kappa.json          4범주 kappa, 4군 대 비4군 kappa(부트스트랩 95% CI), 원일치율, 층별 kappa
      RESULTS_DIR/rater_disagreements.csv  불일치 코드 (합의 회의용, 에너지 비율 순)
      RESULTS_DIR/gold_consensus_draft.csv 일치 코드는 값, 불일치는 빈칸
          → 합의 후 빈칸을 채워 gold_labels.csv로 저장 (열: N_FCODE, nova, uncertain, stratum, 에너지비율_pct)

규칙: kappa는 합의 전 값으로 보고한다. 분류기 예측을 본 뒤 배정을 고치지 않는다.
사용법: python 04_kappa_consensus.py [--a 파일] [--b 파일]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import config as C


def kappa(a, b, labels):
    return float(cohen_kappa_score(a, b, labels=labels))


def boot_ci(a, b, labels, seed=C.SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    ks = []
    for _ in range(C.N_BOOT):
        i = rng.integers(0, len(a), len(a))
        ks.append(cohen_kappa_score(a[i], b[i], labels=labels))
    return [round(float(np.nanpercentile(ks, 2.5)), 3), round(float(np.nanpercentile(ks, 97.5)), 3)]


def load(path):
    df = pd.read_csv(path, dtype={"N_FCODE": str}, encoding="utf-8-sig")
    df["nova"] = pd.to_numeric(df["nova"], errors="coerce")
    df["uncertain"] = pd.to_numeric(df["uncertain"], errors="coerce").fillna(0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=C.RESULTS_DIR / "gold_rater_A.csv")
    ap.add_argument("--b", default=C.RESULTS_DIR / "gold_rater_B.csv")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    m = A.merge(B[["N_FCODE", "nova", "uncertain", "rule", "note"]], on="N_FCODE", suffixes=("_A", "_B"))
    empty = m[m.nova_A.isna() | m.nova_B.isna()]
    if len(empty):
        raise SystemExit(f"nova 빈칸 {len(empty)}개 (규칙 D10: 빈칸 금지): {empty.N_FCODE.head(10).tolist()}")
    if not (m.nova_A.isin([1, 2, 3, 4]).all() and m.nova_B.isin([1, 2, 3, 4]).all()):
        raise SystemExit("nova 값은 1~4만 허용")
    a4, b4 = m.nova_A.astype(int), m.nova_B.astype(int)
    a2, b2 = (a4 == 4).astype(int), (b4 == 4).astype(int)
    rep = {
        "n_codes": int(len(m)),
        "kappa_4cat": round(kappa(a4, b4, [1, 2, 3, 4]), 3), "kappa_4cat_95ci": boot_ci(a4, b4, [1, 2, 3, 4]),
        "agree_4cat": round(float((a4 == b4).mean()), 3),
        "kappa_upf": round(kappa(a2, b2, [0, 1]), 3), "kappa_upf_95ci": boot_ci(a2, b2, [0, 1]),
        "kappa_4cat_by_stratum": {s: round(kappa(a4[m.stratum == s], b4[m.stratum == s], [1, 2, 3, 4]), 3)
                                  for s in ("top", "tail")},
        "uncertain_rate": {"A": round(float(m.uncertain_A.mean()), 3), "B": round(float(m.uncertain_B.mean()), 3)},
    }
    (C.RESULTS_DIR / "rater_kappa.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(pd.crosstab(a4, b4, rownames=["A"], colnames=["B"]))
    dis = m[a4 != b4].sort_values("에너지비율_pct", ascending=False)
    dis.to_csv(C.RESULTS_DIR / "rater_disagreements.csv", index=False, encoding="utf-8-sig")
    draft = m[["순위", "N_FCODE", "식품명", "stratum", "에너지비율_pct"]].copy()
    draft["nova"] = np.where(a4 == b4, a4, np.nan)
    draft["uncertain"] = ((m.uncertain_A == 1) | (m.uncertain_B == 1)).astype(int)
    draft.sort_values("순위").to_csv(C.RESULTS_DIR / "gold_consensus_draft.csv", index=False, encoding="utf-8-sig")
    print(f"불일치 {len(dis)}개 → rater_disagreements.csv / 합의 초안 → gold_consensus_draft.csv")


if __name__ == "__main__":
    main()
