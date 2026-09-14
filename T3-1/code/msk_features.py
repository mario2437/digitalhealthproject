"""MSK-MET (cBioPortal msk_met_2021) 특징 행렬 생성 모듈 (재사용 가능).

다른 팀/분석에서도 그대로 import 해서 쓸 수 있도록 T3-1 분석 고유 설정(코호트 정의,
모델 등)은 넣지 않았다. 이 모듈은 '읽기 + 샘플×특징 행렬 만들기'만 한다.

사용 예:
    import msk_features as mf
    clin = mf.load_clinical()                       # 샘플+환자 임상 병합
    genes = mf.load_common_genes()                  # 341 공통 유전자
    G = mf.build_genomic_matrix(clin.SAMPLE_ID, genes)   # 0/1 변이 지표 (MUT_/AMP_/DEL_/SV_)
    G = mf.filter_by_frequency(G, min_freq=0.01)    # 빈도 필터

특징 정의 (문서화된 결정):
  - MUT_<gene>: 비침묵(non-silent) 체세포 변이. data_mutations.txt 의 mutationType 중
    Missense/Nonsense/Frame_Shift_Del/Ins/In_Frame_Del/Ins/Splice_Site/Translation_Start_Site/
    Nonstop 을 포함하고, TERT 의 5'Flank(프로모터 변이)는 포함한다. Splice_Region 은 제외.
  - AMP_<gene>: CNA 값 2 (고수준 증폭), DEL_<gene>: CNA 값 -2 (심부 결실).
  - SV_<gene>: 구조변이(융합 포함)의 Site1 또는 Site2 유전자가 해당 유전자인 경우.
  - 유전자 이름 별칭: 공통 유전자 목록의 PAK5 는 CNA 행렬에서 PAK7 로 표기되어 있어 맞춰 준다.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("MSK_MET_DIR", str(Path(__file__).resolve().parents[2] / "data" / "msk_met_2021")))

ORGANS = {
    "DMETS_DX_LUNG": "폐",
    "DMETS_DX_LIVER": "간",
    "DMETS_DX_BONE": "뼈",
    "DMETS_DX_CNS_BRAIN": "뇌",
    "DMETS_DX_INTRA_ABDOMINAL": "복강내",
}

NONSILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
}
GENE_ALIAS = {"PAK7": "PAK5"}  # 데이터 표기 -> 공통 유전자 목록 표기


def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, sep="\t", comment="#", low_memory=False, **kw)


def load_clinical() -> pd.DataFrame:
    """샘플 임상 + 환자 임상(SEX, AGE_AT_SEQUENCING 등)을 PATIENT_ID 로 병합."""
    s = _read("data_clinical_sample.txt")
    p = _read("data_clinical_patient.txt")
    df = s.merge(p, on="PATIENT_ID", how="left", validate="many_to_one")
    df["IS_DIST_MET_MAPPED"] = df["IS_DIST_MET_MAPPED"].astype(str).str.upper().eq("TRUE")
    return df


def labels_to_binary(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """'Yes'/'No' 문자열 라벨을 1/0 정수로."""
    cols = list(cols or ORGANS)
    out = pd.DataFrame(index=df.index)
    for c in cols:
        v = df[c].map({"Yes": 1, "No": 0})
        if v.isna().any():
            raise ValueError(f"{c}: Yes/No 외 값 {df.loc[v.isna(), c].unique()[:5]}")
        out[c] = v.astype(int)
    return out


def load_common_genes() -> list[str]:
    path = DATA_DIR / "gene_panels" / "common_genes.txt"
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def build_genomic_matrix(sample_ids, genes, types=("MUT", "AMP", "DEL", "SV")) -> pd.DataFrame:
    """샘플 × (유형_유전자) 0/1 행렬. sample_ids 순서를 index 로 유지한다."""
    sample_ids = pd.Index(pd.unique(pd.Series(list(sample_ids))), name="SAMPLE_ID")
    genes = list(genes)
    gset = set(genes)
    blocks = []

    if "MUT" in types:
        mu = _read("data_mutations.txt", usecols=["sampleId", "hugoGeneSymbol", "mutationType"])
        mu["hugoGeneSymbol"] = mu["hugoGeneSymbol"].replace(GENE_ALIAS)
        keep = mu["mutationType"].isin(NONSILENT) | (
            (mu["mutationType"] == "5'Flank") & (mu["hugoGeneSymbol"] == "TERT"))
        mu = mu[keep & mu["hugoGeneSymbol"].isin(gset) & mu["sampleId"].isin(sample_ids)]
        m = pd.crosstab(mu["sampleId"], mu["hugoGeneSymbol"]).clip(upper=1)
        blocks.append(m.reindex(index=sample_ids, columns=genes, fill_value=0).add_prefix("MUT_"))

    if "AMP" in types or "DEL" in types:
        cna = _read("data_cna.txt", index_col=0)
        cna.index = [GENE_ALIAS.get(g, g) for g in cna.index]
        missing = [s for s in sample_ids if s not in cna.columns]
        if missing:
            raise ValueError(f"CNA 행렬에 없는 샘플 {len(missing)}건")
        cna = cna.reindex(index=genes)[list(sample_ids)].T  # 샘플 × 유전자 (없는 유전자는 NaN)
        cna.index.name = "SAMPLE_ID"
        if "AMP" in types:
            blocks.append((cna == 2).astype(int).add_prefix("AMP_"))
        if "DEL" in types:
            blocks.append((cna == -2).astype(int).add_prefix("DEL_"))

    if "SV" in types:
        sv = _read("data_sv.txt", usecols=["Sample_Id", "Site1_Hugo_Symbol", "Site2_Hugo_Symbol"])
        long = pd.concat([
            sv[["Sample_Id", "Site1_Hugo_Symbol"]].set_axis(["s", "g"], axis=1),
            sv[["Sample_Id", "Site2_Hugo_Symbol"]].set_axis(["s", "g"], axis=1),
        ], ignore_index=True).dropna()
        long["g"] = long["g"].replace(GENE_ALIAS)
        long = long[long["g"].isin(gset) & long["s"].isin(sample_ids)].drop_duplicates()
        m = pd.crosstab(long["s"], long["g"]).clip(upper=1)
        blocks.append(m.reindex(index=sample_ids, columns=genes, fill_value=0).add_prefix("SV_"))

    G = pd.concat(blocks, axis=1).fillna(0).astype(np.int8)
    G.index.name = "SAMPLE_ID"
    return G


def filter_by_frequency(G: pd.DataFrame, min_freq: float = 0.01) -> pd.DataFrame:
    """변이 빈도 min_freq 이상인 특징만 남긴다 (라벨을 쓰지 않는 비지도 필터)."""
    return G.loc[:, G.mean(axis=0) >= min_freq]


def clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """(b) 임상 특징: 나이, 성별, TMB, FGA, MSI 점수/유형, 유전자 패널 (원-핫)."""
    X = pd.DataFrame(index=df.index)
    X["AGE_AT_SEQUENCING"] = pd.to_numeric(df["AGE_AT_SEQUENCING"], errors="coerce")
    X["SEX_MALE"] = df["SEX"].map({"Male": 1, "Female": 0})
    X["TMB_NONSYNONYMOUS"] = df["TMB_NONSYNONYMOUS"].astype(float)
    X["FGA"] = df["FGA"].astype(float)
    X["MSI_SCORE"] = df["MSI_SCORE"].astype(float)
    X = X.join(pd.get_dummies(df["MSI_TYPE"].fillna("NA"), prefix="MSI").astype(int))
    X = X.join(pd.get_dummies(df["GENE_PANEL"], prefix="PANEL").astype(int))
    return X


def one_hot(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(df[col].fillna("NA"), prefix=prefix).astype(int)
