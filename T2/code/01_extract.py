"""1단계: 원시자료에서 필요한 열만 읽어 캐시로 저장한다 (CP949).

출력 (RESULTS_DIR/_cache)
    rc.pkl   24시간 회상 식품 단위 행 (19-59세): 식품코드·식품명·제품명·제조사·식품군·에너지
    all.pkl  개인 가중치(wt_tot)와 연령 (UPF% 모집단 추정용)
"""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyreadstat

import config as C

RC_COLS = ["ID", "year", "age", "N_DNAME", "N_FCODE", "N_FNAME", "N_FD_NS",
           "N_PRODUCT", "N_CNAME", "N_KINDG1", "NF_EN"]
ALL_COLS = ["ID", "year", "age", "sex", "wt_tot"]


def _read(path, cols):
    _, meta = pyreadstat.read_sas7bdat(path, metadataonly=True, encoding="CP949")
    df, _ = pyreadstat.read_sas7bdat(path, usecols=[c for c in cols if c in meta.column_names],
                                     encoding="CP949")
    return df


def read_rc(year):
    df = _read(C.DATA_DIR / f"hn{year % 100:02d}_24rc.sas7bdat", RC_COLS)
    df = df[(df.age >= C.AGE_MIN) & (df.age <= C.AGE_MAX)].copy()
    for c in ["N_FCODE", "N_FNAME", "N_PRODUCT", "N_CNAME", "N_KINDG1", "N_DNAME", "N_FD_NS"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["N_FCODE"] = df["N_FCODE"].str.zfill(5)
    df["N_KINDG1"] = df["N_KINDG1"].str.replace(r"\.0$", "", regex=True).str.zfill(2)
    return df


def read_all(year):
    return _read(C.DATA_DIR / f"hn{year % 100:02d}_all.sas7bdat", ALL_COLS)


def main():
    C.ensure_dirs()
    with ProcessPoolExecutor(min(8, C.N_CPU)) as ex:
        rc = pd.concat(list(ex.map(read_rc, C.YEARS)), ignore_index=True)
        al = pd.concat(list(ex.map(read_all, C.YEARS)), ignore_index=True)
    rc.to_pickle(C.CACHE_DIR / "rc.pkl")
    al.to_pickle(C.CACHE_DIR / "all.pkl")
    print(f"24RC 식품행 {len(rc):,} (19-59세, {rc.ID.nunique():,}명) / 기본자료 {len(al):,}명")


if __name__ == "__main__":
    main()
