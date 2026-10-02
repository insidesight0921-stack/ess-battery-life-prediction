"""
하이퍼파라미터를 더 넓게 탐색하면 복잡한 모델이 Baseline을 이길 수 있는가?

- 기존(model_zoo): 작은 Grid (모델당 4~12 조합)
- 이번: RandomizedSearchCV로 넓은 범위에서 모델당 40 조합, 정책 GroupKFold
- 정책 단위 Hold-out × 10회 (seed 0~9). 선택 기준은 Valid
- 추가로 기록: 튜닝 중 CV 점수(inner) vs 실제 Valid 점수
  → 탐색을 많이 할수록 CV 점수가 '운 좋게 좋은 조합'을 고르게 되어 낙관적이 되는지 확인
"""
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.train import (BASELINE_FEATURES, FEATURES, ROOT, SEED, TARGET, PolicyGroupKFold, load_features,
                       policy_holdout)

warnings.filterwarnings("ignore")
N_ITER = 40


def rs(est, space, cv):
    return RandomizedSearchCV(est, space, n_iter=N_ITER, cv=cv, scoring="neg_mean_squared_error",
                              random_state=SEED, n_jobs=2)


def candidates(cv):
    return {
        "ElasticNet (넓은 탐색)": rs(make_pipeline(StandardScaler(), ElasticNet(max_iter=50_000)),
                                {"elasticnet__alpha": loguniform(1e-5, 1), "elasticnet__l1_ratio": uniform(0, 1)}, cv),
        "SVR (넓은 탐색)": rs(make_pipeline(StandardScaler(), SVR()),
                           {"svr__C": loguniform(1e-2, 1e3), "svr__epsilon": loguniform(1e-3, 1e-1),
                            "svr__gamma": loguniform(1e-3, 1), "svr__kernel": ["rbf", "linear"]}, cv),
        "RandomForest (넓은 탐색)": rs(RandomForestRegressor(random_state=SEED, n_estimators=200),
                                    {"max_depth": [2, 3, 4, 6, None], "min_samples_leaf": randint(1, 8),
                                     "max_features": [0.33, 0.66, 1.0]}, cv),
        "XGBoost (넓은 탐색)": rs(XGBRegressor(random_state=SEED, verbosity=0, n_jobs=1),
                               {"n_estimators": randint(50, 500), "max_depth": randint(1, 5),
                                "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.5, 0.5),
                                "min_child_weight": randint(1, 6), "reg_lambda": loguniform(1e-2, 10)}, cv),
        "LightGBM (넓은 탐색)": rs(LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1),
                                {"n_estimators": randint(50, 500), "num_leaves": randint(2, 16),
                                 "learning_rate": loguniform(0.01, 0.3), "min_child_samples": randint(2, 8),
                                 "subsample": uniform(0.5, 0.5), "subsample_freq": [1],
                                 "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-3, 10)}, cv),
    }


b1 = load_features()["b1"]
cv = PolicyGroupKFold(b1["policy"].to_dict(), n_splits=5)
rows = []
for seed in range(10):
    tr, va = policy_holdout(b1, seed=seed)
    base = make_pipeline(StandardScaler(), LinearRegression())
    inner = cross_val_predict(base, tr[BASELINE_FEATURES], tr[TARGET], cv=cv)
    base.fit(tr[BASELINE_FEATURES], tr[TARGET])
    rows.append([seed, "Baseline", mape(10 ** tr[TARGET], 10 ** inner) * 100,
                 mape(10 ** va[TARGET], 10 ** base.predict(va[BASELINE_FEATURES])) * 100])
    for name, m in candidates(cv).items():
        m.fit(tr[FEATURES], tr[TARGET])
        # 튜닝이 고른 조합의 CV 예측 MAPE (같은 폴드에서 다시 계산)
        inner = cross_val_predict(m.best_estimator_, tr[FEATURES], tr[TARGET], cv=cv)
        rows.append([seed, name, mape(10 ** tr[TARGET], 10 ** inner) * 100,
                     mape(10 ** va[TARGET], 10 ** m.predict(va[FEATURES])) * 100])
    print("seed", seed, "done", flush=True)

df = pd.DataFrame(rows, columns=["seed", "model", "inner_cv_mape", "valid_mape"])
wide = df.pivot(index="seed", columns="model", values="valid_mape")
s = df.groupby("model").agg(CV점수=("inner_cv_mape", "mean"), Valid평균=("valid_mape", "mean"),
                            Valid표준편차=("valid_mape", "std"))
s["낙관 정도(Valid−CV)"] = s["Valid평균"] - s["CV점수"]
s["Baseline보다 나은 횟수"] = wide.lt(wide["Baseline"], axis=0).sum()
print("\n=== 넓은 하이퍼파라미터 탐색 결과 (정책 단위 Hold-out × 10회) ===")
print(s.sort_values("Valid평균").round(2).to_string())
df.to_csv(ROOT / "results" / "tuning_deep_repeated_holdout.csv", index=False)
s.sort_values("Valid평균").round(2).to_csv(ROOT / "results" / "tuning_deep_summary.csv")
