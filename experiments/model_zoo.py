"""
DAY1 후보에 없던 모델 추가 비교 — 선택은 Batch1 Valid로만 (Test는 참고용으로 마지막에 한 번 표시)

같은 조건: 같은 피처 6개(또는 1개), log(수명) 타깃, 정책 단위 Hold-out × 20회, Train 안에서 정책 GroupKFold 튜닝
추가 모델과 넣은 이유
- 선형 계열: LinearRegression(6피처, 규제 없음), Ridge(L2), Lasso(L1), Huber(이상치에 강한 손실)
  → "규제/손실 함수만 바꿔도 달라지나?"
- 커널·거리 기반: SVR(RBF), KNN, Gaussian Process
  → 적은 데이터에서 자주 쓰는 비선형 모델. GPR은 배터리 수명 연구에서 흔히 쓰임
- 트리 앙상블: RandomForest(Bagging), XGBoost(Boosting) → LightGBM 결과가 트리 공통 문제인지 확인
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, DotProduct
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.train import (BASELINE_FEATURES, FEATURES, ROOT, SEED, TARGET, PolicyGroupKFold, load_features,
                       policy_holdout)

warnings.filterwarnings("ignore")


def gs(est, grid, cv):
    return GridSearchCV(est, grid, cv=cv, scoring="neg_mean_squared_error")


def zoo(cv):
    sc = StandardScaler
    return {
        # 기준
        "Baseline (선형, dQ_log_var 1개)": (make_pipeline(sc(), LinearRegression()), BASELINE_FEATURES),
        # 선형 계열 (6피처)
        "Linear (6피처, 규제 없음)": (make_pipeline(sc(), LinearRegression()), FEATURES),
        "Ridge": (gs(make_pipeline(sc(), Ridge()), {"ridge__alpha": np.logspace(-3, 2, 11)}, cv), FEATURES),
        "Lasso": (gs(make_pipeline(sc(), Lasso(max_iter=50_000)), {"lasso__alpha": np.logspace(-4, -1, 10)}, cv), FEATURES),
        "Huber": (gs(make_pipeline(sc(), HuberRegressor(max_iter=2000)),
                     {"huberregressor__epsilon": [1.35, 2.0], "huberregressor__alpha": [1e-4, 1e-2, 1]}, cv), FEATURES),
        # 커널·거리 기반
        "SVR (RBF)": (gs(make_pipeline(sc(), SVR()), {"svr__C": [0.1, 1, 10], "svr__epsilon": [0.01, 0.05],
                                                     "svr__gamma": ["scale", 0.05]}, cv), FEATURES),
        "KNN": (gs(make_pipeline(sc(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 5, 7],
                                                              "kneighborsregressor__weights": ["uniform", "distance"]}, cv), FEATURES),
        "Gaussian Process": (make_pipeline(sc(), GaussianProcessRegressor(
            kernel=ConstantKernel() * DotProduct() + ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(),
            normalize_y=True, random_state=SEED)), FEATURES),
        # 트리 앙상블
        "RandomForest": (gs(RandomForestRegressor(random_state=SEED, n_estimators=200, n_jobs=-1),
                            {"max_depth": [2, 4, None], "min_samples_leaf": [2, 4]}, cv), FEATURES),
        "XGBoost": (gs(XGBRegressor(random_state=SEED, n_estimators=200, verbosity=0, subsample=0.8, n_jobs=2),
                       {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1]}, cv), FEATURES),
    }


data = load_features()
b1 = data["b1"]
cv = PolicyGroupKFold(b1["policy"].to_dict(), n_splits=5)

rows = []
for seed in range(20):
    tr, va = policy_holdout(b1, seed=seed)
    for name, (model, feats) in zoo(cv).items():
        model.fit(tr[feats], tr[TARGET])
        rows.append([seed, name, mape(10 ** va[TARGET], 10 ** model.predict(va[feats])) * 100])
df = pd.DataFrame(rows, columns=["seed", "model", "valid_mape"])
wide = df.pivot(index="seed", columns="model", values="valid_mape")
base = wide["Baseline (선형, dQ_log_var 1개)"]
summary = pd.DataFrame({"Valid 평균": wide.mean(), "표준편차": wide.std(),
                        "Baseline보다 나은 횟수": (wide.lt(base, axis=0)).sum()})

# 참고용 Test: train.py와 같은 기본 분할(seed=42)로 한 번만 (선택에는 사용하지 않음)
tr, va = policy_holdout(b1)
ref = {}
for name, (model, feats) in zoo(cv).items():
    model.fit(tr[feats], tr[TARGET])
    p2, p3 = (10 ** model.predict(data[b][feats]) for b in ["b2", "b3"])
    ref[name] = [mape(data["b2"]["cycle_life"], p2) * 100, mape(data["b3"]["cycle_life"], p3) * 100, p3.max()]
summary[["(참고) Test B2", "(참고) Test B3", "(참고) B3 예측 최대"]] = pd.DataFrame(ref, index=["b2", "b3", "mx"]).T.values \
    if False else pd.DataFrame(ref).T.reindex(summary.index).values
print("=== 추가 모델 비교: Valid MAPE (정책 단위 Hold-out × 20회) — 선택 기준 ===")
print(summary.sort_values("Valid 평균").round(2).to_string())
df.to_csv(ROOT / "results" / "model_zoo_repeated_holdout.csv", index=False)
summary.sort_values("Valid 평균").round(2).to_csv(ROOT / "results" / "model_zoo_summary.csv")
