"""2단계: 식품코드 단위 표를 만든다. 분류기의 입력(식품명·식품군·제품명·제조사)과 에너지 기여를 담는다.

출력 RESULTS_DIR/foodcode_list.csv
    N_FCODE, 식품명, 식품군코드, 식품군명, 섭취건수, 에너지_kcal, 에너지비율_pct, 누적에너지비율_pct, 순위,
    제품정보_기재비율, 대표_제품명, 대표_제조사 (분류기 입력; 건수 괄호 포함),
    대표_응답식품명, 대표_음식명 (사람 배정자 참고용; 분류기 입력에서는 제외)
"""
from collections import Counter

import pandas as pd

import config as C


def top_values(s, k=3, skip=("", "888")):
    cnt = Counter(v for v in s if v not in skip)
    return " / ".join(f"{v}({n})" for v, n in cnt.most_common(k))


def main():
    C.ensure_dirs()
    rc = pd.read_pickle(C.CACHE_DIR / "rc.pkl")
    rc["has_product"] = rc.N_PRODUCT.ne("")   # 888(가공식품이나 제품명 모름)도 기재로 센다
    g = rc.groupby("N_FCODE")
    tab = pd.DataFrame({
        "식품명": g.N_FNAME.agg(lambda s: s.value_counts().index[0]),
        "식품군코드": g.N_KINDG1.agg(lambda s: s.value_counts().index[0]),
        "섭취건수": g.size(),
        "에너지_kcal": g.NF_EN.sum().round(1),
        "제품정보_기재비율": g.has_product.mean().round(3),
        "대표_제품명": g.N_PRODUCT.agg(top_values),
        "대표_제조사": g.N_CNAME.agg(top_values),
        "대표_응답식품명": g.N_FD_NS.agg(top_values),
        "대표_음식명": g.N_DNAME.agg(top_values),
    })
    tab.insert(2, "식품군명", tab["식품군코드"].map(C.KINDG))
    tab = tab.sort_values(["에너지_kcal", "섭취건수"], ascending=False)
    tab["에너지비율_pct"] = 100 * tab["에너지_kcal"] / tab["에너지_kcal"].sum()
    tab["누적에너지비율_pct"] = tab["에너지비율_pct"].cumsum()
    tab["순위"] = range(1, len(tab) + 1)
    tab = tab.reset_index()
    tab.to_csv(C.RESULTS_DIR / "foodcode_list.csv", index=False, encoding="utf-8-sig")
    cov = {p: int((tab["누적에너지비율_pct"].shift(fill_value=0) < p).sum()) for p in (80, 90, 95)}
    print(f"식품코드 {len(tab):,}개 / 누적 에너지 80·90·95%를 채우는 코드 수: {cov}")


if __name__ == "__main__":
    main()
