"""3단계: 정답셋(평가셋) 표본을 뽑고 2인 독립 배정 템플릿을 만든다.

표본 설계 (SAP 3절, 시드 고정)
    상위층: 에너지 기여 상위 GOLD_TOP_N(200)개 코드 전수
    꼬리층: 나머지 코드에서 식품군 비례 층화 무작위 GOLD_TAIL_N(200)개
    → 두 층을 합친 400개가 정답셋이다. 분류기 규칙·프롬프트 개발에 이 코드들을 쓰지 않는다.
    보정 연습용 30개와 LLM few-shot 예시 후보는 표본 밖 코드에서만 고른다.

출력 (RESULTS_DIR)
    gold_sample.csv          표본 코드 목록 (stratum: top/tail, 표본추출 가중치 w_sample)
    practice_codes.csv       보정 연습용 30개 (표본 밖, kappa에 포함하지 않음)
    gold_rater_A.csv, gold_rater_B.csv
                             배정 템플릿. 분류기 예측은 담지 않는다(맹검). 이미 있으면 덮어쓰지 않는다.
"""
import numpy as np
import pandas as pd

import config as C

SHOW = ["순위", "N_FCODE", "식품명", "식품군명", "에너지비율_pct", "제품정보_기재비율",
        "대표_제품명", "대표_제조사", "대표_응답식품명", "대표_음식명"]


def proportional_alloc(sizes, n):
    raw = sizes / sizes.sum() * n
    base = np.floor(raw).astype(int)
    rest = n - base.sum()
    order = (raw - base).sort_values(ascending=False).index[:rest]
    base[order] += 1
    return base


def main():
    rng = np.random.default_rng(C.SEED)
    t = pd.read_csv(C.RESULTS_DIR / "foodcode_list.csv", dtype={"N_FCODE": str, "식품군코드": str},
                    encoding="utf-8-sig")
    top = t[t["순위"] <= C.GOLD_TOP_N].copy()
    top["stratum"], top["w_sample"] = "top", 1.0
    rest = t[t["순위"] > C.GOLD_TOP_N]
    alloc = proportional_alloc(rest.groupby("식품군코드").size(), C.GOLD_TAIL_N)
    parts = []
    for kg, n in alloc.items():
        pool = rest[rest["식품군코드"] == kg]
        if n > 0:
            idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
            parts.append(t.loc[np.sort(idx)])
    tail = pd.concat(parts)
    tail["stratum"], tail["w_sample"] = "tail", len(rest) / C.GOLD_TAIL_N
    gold = pd.concat([top, tail]).sort_values("순위")
    gold.to_csv(C.RESULTS_DIR / "gold_sample.csv", index=False, encoding="utf-8-sig")

    outside = t[~t.N_FCODE.isin(gold.N_FCODE) & (t["순위"] <= 1000)]
    prac = outside.sample(30, random_state=C.SEED % (2**32)).sort_values("순위")
    prac[SHOW].assign(nova="", uncertain="", rule="", note="").to_csv(
        C.RESULTS_DIR / "practice_codes.csv", index=False, encoding="utf-8-sig")

    tmpl = gold[SHOW + ["stratum"]].copy()
    # 배정 순서를 무작위로 섞어 에너지 순위에 따른 판단 습관을 줄인다
    tmpl = tmpl.sample(frac=1, random_state=C.SEED % (2**32))
    tmpl["nova"], tmpl["uncertain"], tmpl["rule"], tmpl["note"] = "", "", "", ""
    for r in ("A", "B"):
        out = C.RESULTS_DIR / f"gold_rater_{r}.csv"
        if out.exists():
            print(f"[건너뜀] {out.name} 이미 있음")
            continue
        tmpl.to_csv(out, index=False, encoding="utf-8-sig")
    e_top, e_tail = top["에너지비율_pct"].sum(), tail["에너지비율_pct"].sum()
    print(f"정답셋 {len(gold)}개 (상위 {len(top)}: 에너지 {e_top:.1f}%, 꼬리 {len(tail)}: 에너지 {e_tail:.2f}%)")
    print("꼬리층 식품군별 배정:", alloc[alloc > 0].to_dict())


if __name__ == "__main__":
    main()
