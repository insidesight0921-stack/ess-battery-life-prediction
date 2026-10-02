"""
최종 모델 선택 — Test(Batch2·3)는 보지 않고 Batch1 안에서만 결정

이유: 단일 Hold-out 결과는 분할 운에 따라 ±2%p 흔들림 (split_leakage_check 참고)
방법: 정책 단위 Hold-out을 seed 20개로 반복, 매번 Train 부분에서 튜닝(정책 GroupKFold) 후 Valid MAPE 기록
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape

from src.features import select_features
from src.train import (ROOT, TARGET, PolicyGroupKFold, build_models, load_features, make_voting,
                       policy_holdout)

warnings.filterwarnings("ignore")
b1 = load_features()["b1"]
cv = PolicyGroupKFold(b1["policy"].to_dict(), n_splits=5)
rows, feat_log = [], []
for seed in range(20):
    tr, va = policy_holdout(b1, seed=seed)
    fitted = {}
    sel = select_features(tr)                      # 매 반복마다 Train 부분에서만 피처 선택
    feat_log.append([seed, len(sel), ",".join(sel)])
    for name, (model, feats) in build_models(cv, sel).items():
        if model == "voting":
            model = make_voting(fitted["ElasticNet"], fitted["LightGBM"])
        model.fit(tr[feats], tr[TARGET])
        fitted[name] = model
        rows.append([seed, name, mape(10 ** va[TARGET], 10 ** model.predict(va[feats])) * 100])
df = pd.DataFrame(rows, columns=["seed", "model", "valid_mape"])
wide = df.pivot(index="seed", columns="model", values="valid_mape")
summary = pd.DataFrame({"평균": wide.mean(), "표준편차": wide.std(), "최악": wide.max(),
                        "1등 횟수": wide.idxmin(axis=1).value_counts().reindex(wide.columns).fillna(0).astype(int)})
print("=== Valid MAPE (정책 단위 Hold-out × 20회) ===")
print(summary.sort_values("평균").round(2).to_string())
base = wide["Baseline (Variance)"]
for m in wide.columns.drop("Baseline (Variance)"):
    d = wide[m] - base
    print(f"  {m:18s} - Baseline: 평균 {d.mean():+.2f}%p, Baseline보다 나은 횟수 {(d < 0).sum()}/20")
df.to_csv(ROOT / "results" / "model_selection_repeated_holdout.csv", index=False)
fl = pd.DataFrame(feat_log, columns=["seed", "n_features", "features"])
fl.to_csv(ROOT / "results" / "feature_selection_by_seed.csv", index=False)
print("\n=== 반복마다 선택된 피처 (Train 기준) ===")
print("피처별 선택 횟수 /20:", pd.Series(",".join(fl.features).split(",")).value_counts().to_dict())
