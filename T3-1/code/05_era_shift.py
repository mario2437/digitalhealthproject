"""05. H3: 흔적을 쓴 모델은 검사 시기가 바뀌면 더 크게 무너지는가.

주 분할(temporal): PID_NUM 오름차순 앞 70% 학습 / 뒤 30% 시험.
보조 분할(cross-panel): IMPACT468 학습 → IMPACT341+410 시험, 역방향 (학습 내 패널 상수 → A에서 PANEL 제외).
모델: T(참고), B, B+A.  내부 추정 = 학습 세트 5-겹 OOF AUROC, 외부 = 학습 전체로 적합 후 시험 AUROC.
격차 = 내부 − 외부.  격차 차이 = 격차(B+A) − 격차(B), 학습 OOF·시험 예측을 독립 부트스트랩.
H3 기준: 격차 차이 >= 0.02 인 장기 >= 2/5.
보조 지표: 시험 세트 O/E(평균 예측확률 / 관찰 유병률).
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from t31_common import (N_BOOT, N_JOBS, ORGAN_KEYS, OUTCOMES, RESULTS, SEED, TRAIN_FRAC, design, fast_auc,
                        make_lgbm, oof_predict)

b = pd.read_pickle(RESULTS / "features_primary.pkl")
pid = b["A"]["PID_NUM"].values
panel = b["meta"]["GENE_PANEL"].values
cut = np.quantile(pid, TRAIN_FRAC)
SPLITS = {
    "temporal": (pid <= cut, pid > cut, False),
    "468→341/410": (panel == "IMPACT468", panel != "IMPACT468", True),
    "341/410→468": (panel != "IMPACT468", panel == "IMPACT468", True),
}


def gap_boot(y_tr, s_tr_new, s_tr_ref, y_te, s_te_new, s_te_ref, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y_tr), len(y_tr))
        j = rng.integers(0, len(y_te), len(y_te))
        if y_te[j].min() == y_te[j].max():
            continue
        g_new = fast_auc(y_tr[i], s_tr_new[i]) - fast_auc(y_te[j], s_te_new[j])
        g_ref = fast_auc(y_tr[i], s_tr_ref[i]) - fast_auc(y_te[j], s_te_ref[j])
        out.append(g_new - g_ref)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


rows, preds, jobs = [], [], []
for split, (tr, te, drop_panel) in SPLITS.items():
    info = dict(split=split, n_train=int(tr.sum()), n_test=int(te.sum()))
    pr = pd.DataFrame({"SAMPLE_ID": b["meta"]["SAMPLE_ID"].values[te]})
    for o in OUTCOMES:
        y = b["Y"][o].values
        s = {}
        for sp in ["T", "B", "B+A"]:
            X = design(b, sp, drop_panel=drop_panel)
            X = X.loc[:, X.iloc[np.where(tr)[0]].nunique() > 1]  # 학습 세트에서 상수인 열 제거
            s_tr = oof_predict(X.iloc[np.where(tr)[0]], y[tr])
            s_te = make_lgbm().fit(X.values[tr], y[tr]).predict_proba(X.values[te])[:, 1]
            s[sp] = (s_tr, s_te)
            pr[f"{o}|{sp}"] = s_te
            rows.append(dict(**info, outcome=o, 결과=OUTCOMES[o], model=sp, n_features=X.shape[1],
                             auroc_internal=fast_auc(y[tr], s_tr), auroc_external=fast_auc(y[te], s_te),
                             prev_train=y[tr].mean(), prev_test=y[te].mean(), OE_test=s_te.mean() / y[te].mean()))
            print(split, o, sp, round(rows[-1]["auroc_internal"], 4), round(rows[-1]["auroc_external"], 4), flush=True)
        jobs.append((split, o, (y[tr], s["B+A"][0], s["B"][0], y[te], s["B+A"][1], s["B"][1])))
    preds.append(pr.assign(split=split))

perf = pd.DataFrame(rows)
perf["gap"] = perf.auroc_internal - perf.auroc_external
perf.to_csv(RESULTS / "h3_perf.tsv", sep="\t", index=False, float_format="%.4f")
pd.concat(preds).to_csv(RESULTS / "h3_test_predictions.tsv", sep="\t", index=False, float_format="%.5f")

cis = Parallel(n_jobs=N_JOBS)(delayed(gap_boot)(*args, seed=SEED) for _, _, args in jobs)
gd = []
for (split, o, _), (lo, hi) in zip(jobs, cis):
    p = perf[(perf.split == split) & (perf.outcome == o)].set_index("model")
    gd.append(dict(split=split, outcome=o, 결과=OUTCOMES[o],
                   gap_B=p.loc["B", "gap"], gap_BA=p.loc["B+A", "gap"],
                   gap_diff=p.loc["B+A", "gap"] - p.loc["B", "gap"], ci_lo=lo, ci_hi=hi,
                   internal_gain_A=p.loc["B+A", "auroc_internal"] - p.loc["B", "auroc_internal"],
                   external_gain_A=p.loc["B+A", "auroc_external"] - p.loc["B", "auroc_external"]))
gd = pd.DataFrame(gd)
gd.to_csv(RESULTS / "h3_gap_diff.tsv", sep="\t", index=False, float_format="%.4f")
main = gd[(gd.split == "temporal") & gd.outcome.isin(ORGAN_KEYS)]
summary = {"시간 분할 PID_NUM 기준값": float(cut),
           "시험 세트 패널 구성": pd.Series(panel[SPLITS["temporal"][1]]).value_counts().to_dict(),
           "H3 충족 장기 수 (격차 차이>=0.02)": int((main.gap_diff >= 0.02).sum()),
           "H3 판정 (>=2/5)": bool((main.gap_diff >= 0.02).sum() >= 2),
           "보조: CI 하한>0 장기 수": int((main.ci_lo > 0).sum())}
(RESULTS / "h3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(perf.round(3).to_string())
print(gd.round(4).to_string())
print(summary)
