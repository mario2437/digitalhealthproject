"""T2(v2) 공통 설정: 한국어 식품명 기반 NOVA 자동 분류.

경로는 환경변수로 바꿀 수 있다 (기본값은 code/ 기준 ../results, ../figures).
    T2_DATA_DIR     국민건강영양조사 SAS 원시자료 폴더
    T2_RESULTS_DIR  결과(csv/json/중간파일) 저장 폴더
    T2_FIGURES_DIR  그림 저장 폴더
    T2_GOLD         정답셋 CSV (열: N_FCODE, nova, uncertain). 기본값 RESULTS_DIR/gold_labels.csv
    T2_LLM_MODELS   쉼표로 구분한 Hugging Face 모델 ID (기본: Qwen/Qwen2.5-7B-Instruct)
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("T2_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "knhanes" / "sas")))
RESULTS_DIR = Path(os.environ.get("T2_RESULTS_DIR", CODE_DIR.parent / "results")).resolve()
FIGURES_DIR = Path(os.environ.get("T2_FIGURES_DIR", CODE_DIR.parent / "figures")).resolve()
CACHE_DIR = RESULTS_DIR / "_cache"
GOLD = Path(os.environ.get("T2_GOLD", RESULTS_DIR / "gold_labels.csv"))
PRED_DIR = RESULTS_DIR / "predictions"

YEARS = [2019, 2020, 2021, 2022]      # 제8기 + 제9기 1차년도 (식품성분표 제10개정판, 코드 체계 동일)
AGE_MIN, AGE_MAX = 19, 59
YOUNG_MAX = 39
ENERGY_MIN, ENERGY_MAX = 500, 5000    # 개인 UPF% 계산 대상의 총 에너지 범위
SEED = 20260914

# 정답셋 표본 설계 (SAP 3절)
GOLD_TOP_N = 200        # 에너지 기여 상위 200개 코드는 전수 포함
GOLD_TAIL_N = 200       # 나머지 코드에서 식품군 비례 층화 무작위 200개
N_BOOT = 2000
CV_FOLDS = 5

LLM_MODELS = [m for m in os.environ.get("T2_LLM_MODELS", "Qwen/Qwen2.5-7B-Instruct").split(",") if m]
N_CPU = int(os.environ.get("T2_NCPU", "16"))

KINDG = {"01": "곡류", "02": "감자·전분류", "03": "당류", "04": "두류", "05": "종실류",
         "06": "채소류", "07": "버섯류", "08": "과일류", "09": "해조류", "11": "양념류",
         "12": "유지류(식물)", "13": "기타(식물)", "15": "육류", "16": "난류", "17": "어패류",
         "18": "우유류", "19": "유지류(동물)", "20": "기타(동물)", "21": "음료류", "22": "주류"}


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, CACHE_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
