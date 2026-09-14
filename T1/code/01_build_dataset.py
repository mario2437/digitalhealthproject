"""01. KNHANES 2016-2024 원시자료 읽기 → 변수 조화 → 제외기준 → 연도별 코호트 흐름표.

산출: results/base_cohort.pkl  (제외 1-4 적용 자료 + 미진단 당뇨병·완전자료 표시 열)
      results/cohort_flow.csv   (연도별 제외 단계 인원; 결과변수 값은 사용하지 않음)
"""
import numpy as np
import pandas as pd
import pyreadstat

from common import ALL_YEARS, DATA_DIR, PRED, hba1c_harmonized, out_dirs

RES, _ = out_dirs()

COLS = ["year", "age", "sex", "psu", "kstrata", "wt_itvex", "HE_fst", "HE_prg",
        "DE1_dg", "DI1_dg", "DI2_dg", "DE1_31", "DE1_32", "DI1_2", "DI2_2",
        "HE_glu", "HE_HbA1c", "HE_TG", "HE_HDL_st2",
        "HE_BMI", "HE_wc", "HE_sbp", "HE_dbp",
        "pa_aerobic", "BE8_1", "BE8_2",
        "Total_slp_wk", "Total_slp_wd", "BP16_1", "BP16_2",
        "BP16_11", "BP16_12", "BP16_13", "BP16_14", "BP16_21", "BP16_22", "BP16_23", "BP16_24",
        "BS1_1", "BS3_1", "BD1", "BD1_11", "HE_DMfh1", "HE_DMfh2", "HE_DMfh3"]


def read_year(y):
    path = DATA_DIR / f"hn{str(y)[2:]}_all.sas7bdat"
    _, meta = pyreadstat.read_sas7bdat(str(path), metadataonly=True, encoding="CP949")
    use = [c for c in COLS if c in meta.column_names]
    df, _ = pyreadstat.read_sas7bdat(str(path), usecols=use, encoding="CP949")
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["year"] = y
    return df[COLS]


def valid(s, lo, hi):
    return s.where((s >= lo) & (s <= hi))


def sleep_from_times(d, p):
    """취침·기상 시각(시/분)으로 수면시간(시간). 2016-2018 Total_slp_* 생성 방식과 동일."""
    bh, bm, wh, wm = (d[f"BP16_{p}{k}"] for k in (1, 2, 3, 4))
    ok = (bh <= 24) & (wh <= 24) & (bm <= 59) & (wm <= 59)
    dur = ((wh + wm / 60) - (bh + bm / 60)) % 24
    return dur.where(ok)


def sleep_item_type(y):
    """수면 문항 형식: 'time'(취침·기상 시각) 또는 'hours'(하루 평균 수면시간)."""
    return "hours" if y in (2019, 2020, 2022, 2023) else "time"


def harmonize(d):
    y = int(d["year"].iloc[0])
    o = pd.DataFrame(index=d.index)
    for c in ["year", "age", "psu", "kstrata", "wt_itvex", "HE_fst", "HE_glu", "HE_HbA1c",
              "HE_TG", "HE_HDL_st2", "HE_sbp", "HE_dbp"]:
        o[c] = d[c]
    o["male"] = (d["sex"] == 1).astype(float)
    o["bmi"], o["wc"] = d["HE_BMI"], d["HE_wc"]
    o["sbp"], o["dbp"] = d["HE_sbp"], d["HE_dbp"]
    o["pa_aerobic"] = d["pa_aerobic"].where(d["pa_aerobic"].isin([0, 1]))
    h, m = d["BE8_1"], d["BE8_2"]
    o["sit_hr"] = (valid(h, 0, 24) + valid(m, 0, 59) / 60).where(lambda s: s <= 24)
    if y <= 2018:
        wk, wd = valid(d["Total_slp_wk"], 0, 1440) / 60, valid(d["Total_slp_wd"], 0, 1440) / 60
    elif sleep_item_type(y) == "hours":
        wk, wd = valid(d["BP16_1"], 0, 24), valid(d["BP16_2"], 0, 24)
    else:
        wk, wd = sleep_from_times(d, 1), sleep_from_times(d, 2)
    o["sleep_hr"] = (5 * wk + 2 * wd) / 7
    o["sleep_item"] = sleep_item_type(y)
    smk = pd.Series(np.nan, index=d.index)
    smk[d["BS1_1"].isin([1, 3])] = 0
    smk[(d["BS1_1"] == 2) & (d["BS3_1"] == 3)] = 1
    smk[(d["BS1_1"] == 2) & d["BS3_1"].isin([1, 2])] = 2
    o["smk_current"] = (smk == 2).astype(float).where(smk.notna())
    o["smk_former"] = (smk == 1).astype(float).where(smk.notna())
    drk = pd.Series(np.nan, index=d.index)
    drk[(d["BD1"] == 1) | d["BD1_11"].isin([1, 8])] = 0
    drk[d["BD1_11"].isin([2, 3, 4])] = 1
    drk[d["BD1_11"].isin([5, 6])] = 2
    o["drk_monthly"] = (drk == 1).astype(float).where(drk.notna())
    o["drk_weekly2"] = (drk == 2).astype(float).where(drk.notna())
    fh = d[["HE_DMfh1", "HE_DMfh2", "HE_DMfh3"]]
    o["fh_dm"] = (fh == 1).any(axis=1).astype(float).where(fh.notna().any(axis=1))
    dx = ((d["DE1_dg"] == 1) | (d["DI1_dg"] == 1) | (d["DI2_dg"] == 1) |
          (d["DE1_31"] == 1) | (d["DE1_32"] == 1) |
          d["DI1_2"].isin([1, 2, 3, 4]) | d["DI2_2"].isin([1, 2, 3, 4]))
    dx_unknown = ~d["DE1_dg"].isin([0, 1]) | ~d["DI1_dg"].isin([0, 1]) | ~d["DI2_dg"].isin([0, 1])
    o["ex_noresp"] = dx_unknown & ~dx
    o["ex_dx"] = dx
    o["ex_prg"] = d["HE_prg"] == 1
    o["ex_outcome"] = d["HE_glu"].isna() | d["HE_HbA1c"].isna() | ~(d["HE_fst"] >= 8)
    return o


def main():
    frames = []
    for y in ALL_YEARS:
        d = read_year(y)
        n_all = len(d)
        d = d[(d["age"] >= 19) & (d["age"] <= 64)].reset_index(drop=True)
        h = harmonize(d)
        h["n_raw_file"] = n_all
        frames.append(h)
    allp = pd.concat(frames, ignore_index=True)

    flow, keep = [], []
    for y, d in allp.groupby("year"):
        n0 = len(d)
        s0 = d[~d["ex_noresp"]]
        s1 = s0[~s0["ex_dx"]]
        s2 = s1[~s1["ex_prg"]]
        s3 = s2[~s2["ex_outcome"]].copy()
        s3["undx_harm"] = (s3["HE_glu"] >= 126) | (hba1c_harmonized(s3) >= 6.5)
        s3["undx_raw"] = (s3["HE_glu"] >= 126) | (s3["HE_HbA1c"] >= 6.5)
        s3["cc"] = s3[PRED].notna().all(axis=1)
        s4 = s3[~s3["undx_harm"]]
        s5 = s4[s4["cc"]]
        naive = s3[~s3["undx_raw"] & s3["cc"]]
        flow.append(dict(year=y, file_n=int(d["n_raw_file"].iloc[0]), age19_64=n0,
                         excl_survey_nonresponse=n0 - len(s0), excl_dx_or_treated=len(s0) - len(s1),
                         excl_pregnant=len(s1) - len(s2),
                         excl_no_outcome_or_fasting_lt8h=len(s2) - len(s3),
                         excl_undiagnosed_dm=len(s3) - len(s4),
                         excl_missing_predictors=len(s4) - len(s5), final=len(s5),
                         final_naive_cohort=len(naive),
                         pct_fasting_or_lab_missing=(len(s2) - len(s3)) / max(len(s2), 1),
                         pct_pred_missing=(len(s4) - len(s5)) / max(len(s4), 1)))
        keep.append(s3)
    base = pd.concat(keep, ignore_index=True)
    base.to_pickle(RES / "base_cohort.pkl")
    flow = pd.DataFrame(flow)
    flow.to_csv(RES / "cohort_flow.csv", index=False)
    print(flow.T.to_string())


if __name__ == "__main__":
    main()
