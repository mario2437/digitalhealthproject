"""6단계: 텍스트 머신러닝 분류기 (ML-LR, ML-SVM, ML-GBM). SAP 4절 설정 그대로.

특징: 식품명 + 대표 제품명 + 대표 제조사 문자열의 문자 n-gram(1-3, char_wb) TF-IDF
      + 식품군·코드 첫 자리 원-핫 + 제품정보 기재 비율
평가: 정답셋에서 층화 5겹 교차검증의 out-of-fold 예측 (TF-IDF도 폴드 안에서 학습)
      LR·SVM은 내부 3겹 CV(macro-F1)로 C∈{0.3,1,3,10}을 고른다. GBM은 고정 설정.

입력  RESULTS_DIR/foodcode_list.csv, GOLD (N_FCODE, nova)
출력  PRED_DIR/pred_ML-LR.csv, pred_ML-SVM.csv, pred_ML-GBM.csv (N_FCODE, pred, fold)
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

import config as C
from prompts import strip_counts

os.environ.setdefault("OMP_NUM_THREADS", str(C.N_CPU))


def load_xy():
    t = pd.read_csv(C.RESULTS_DIR / "foodcode_list.csv", dtype={"N_FCODE": str, "식품군코드": str},
                    encoding="utf-8-sig")
    gold = pd.read_csv(C.GOLD, dtype={"N_FCODE": str}, encoding="utf-8-sig")[["N_FCODE", "nova"]]
    d = gold.merge(t, on="N_FCODE", how="left")
    d["text"] = (d["식품명"].fillna("") + " | " + d["대표_제품명"].map(strip_counts) + " | "
                 + d["대표_제조사"].map(strip_counts))
    d["code1"] = d.N_FCODE.str[0]
    d["prod_ratio"] = d["제품정보_기재비율"].fillna(0)
    return d, d["nova"].astype(int).to_numpy()


def features(max_features=None):
    return ColumnTransformer([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True,
                                  min_df=1, max_features=max_features), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["식품군코드", "code1"]),
        ("num", "passthrough", ["prod_ratio"]),
    ])


def models(seed):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    grid = {"clf__C": [0.3, 1, 3, 10]}
    lr = GridSearchCV(Pipeline([("feat", features()),
                                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
                      grid, cv=inner, scoring="f1_macro", n_jobs=1)
    svm = GridSearchCV(Pipeline([("feat", features()),
                                 ("clf", LinearSVC(class_weight="balanced", max_iter=20000))]),
                       grid, cv=inner, scoring="f1_macro", n_jobs=1)
    from lightgbm import LGBMClassifier
    gbm = Pipeline([("feat", features(max_features=2000)),
                    ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                           min_child_samples=5, class_weight="balanced",
                                           random_state=seed, n_jobs=min(16, C.N_CPU), verbose=-1))])
    return {"ML-LR": lr, "ML-SVM": svm, "ML-GBM": gbm}


def main():
    C.ensure_dirs()
    d, y = load_xy()
    X = d[["text", "식품군코드", "code1", "prod_ratio"]]
    outer = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    for name, model in models(C.SEED).items():
        pred, fold, chosen = np.zeros(len(y), int), np.zeros(len(y), int), []
        for k, (tr, te) in enumerate(outer.split(X, y)):
            model.fit(X.iloc[tr], y[tr])
            pred[te] = model.predict(X.iloc[te])
            fold[te] = k
            if hasattr(model, "best_params_"):
                chosen.append(model.best_params_["clf__C"])
        out = pd.DataFrame({"N_FCODE": d.N_FCODE, "pred": pred, "fold": fold})
        out.to_csv(C.PRED_DIR / f"pred_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: OOF 예측 저장 (폴드별 선택 C: {chosen or '고정'})")


if __name__ == "__main__":
    main()
