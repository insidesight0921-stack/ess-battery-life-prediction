"""
학습·평가 모듈: 피처 테이블 → 모델 학습 → 성능 리포트

입력: data/processed/features_{b1,b2,b3}.csv  (src/features.py 결과)
출력: results/model_performance.csv, results/predictions.csv

설계 (DAY1 모델 전략)
- 태스크: 회귀, 타깃 log10(cycle_life)
  · Train(Batch1)의 550 미만 셀이 1/36뿐 → 분류 학습 불가
  · 수명 범위가 4배 이상 차이(392~1,935) → log 변환, 오차를 비율로 다룸(MAPE와 정합)
- 피처: Train 상관 |r| ≥ 0.4 + 서로 |r| > 0.85인 쌍은 하나만 + 도메인 근거 (Test 정보 미사용)
- 모델
  ① Baseline  : dQ_log_var 1개 + 선형회귀 (원 논문 Variance 모델)
  ② ElasticNet: 외삽 가능(Test 수명이 Train 범위 밖) + 다중공선성 대응(L1+L2 규제)
  ③ LightGBM  : 비선형 관계 포착 (단, 외삽 불가)
  ④ Voting    : ②+③ 예측 평균 → 최종 모델
- 검증: Batch1을 충전 정책(policy) 단위로 Train 80% / Valid 20%(Hold-out) 분할, Train 안에서 정책 단위 5-fold CV
  · 같은 정책 셀(대부분 2개씩)이 Train/Valid에 갈라지면 누수 → 정책 단위 분할 (Notion 설명과 같은 이유)
  · v1(무작위 분할)에서는 Valid 8셀 중 5셀이 같은 정책의 짝 셀을 Train에 두고 있었음
- 평가: MAPE(%), RMSE(cycle) — log 예측을 10**pred로 되돌린 원래 단위에서 계산

사용 예 (프로젝트 루트에서):
    python -m src.train
"""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import select_features
from src.preprocess import PROC_DIR, ROOT

warnings.filterwarnings("ignore")

RESULT_DIR = ROOT / "results"
SEED = 42
TARGET = "log_cycle_life"
TARGET_PAPER_MAPE = 9.1   # 원 논문 목표 (%)

# 피처: src.features.select_features로 Train 부분에서만 선택 (규칙: |r|>=0.4, 서로 |r|>0.85면 하나만)
#   v2까지는 DAY1에 손으로 고른 6개를 고정해 썼는데, 규칙을 코드로 적용해 보니 불일치 발견
#   (dQ_at_2V는 dQ_log_var와 0.88로 규칙 위반, QD_slope_91_100 대신 QD_c100_minus_c2가 맞음) → 규칙을 코드로 고정
BASELINE_FEATURES = ["dQ_log_var"]

# 최종 모델: Batch1 안에서만 결정 (experiments/model_selection.py, 정책 단위 Hold-out 20회 반복)
#   Valid MAPE 평균 Baseline 9.55% < ElasticNet 11.11% < Voting 12.15% < LightGBM 13.67%, Baseline 1등 14/20회
FINAL_MODEL = "Baseline (Variance)"


# ---------------------------------------------------------------- 데이터
def load_features():
    return {b: pd.read_csv(PROC_DIR / f"features_{b}.csv", index_col=0) for b in ["b1", "b2", "b3"]}


# ---------------------------------------------------------------- 분할
class PolicyGroupKFold(BaseCrossValidator):
    """X의 index(cell_id) → policy로 묶어 GroupKFold. GridSearchCV 안에서도 그룹이 유지되도록 X에서 직접 읽음"""
    def __init__(self, policy_map, n_splits=5):
        self.policy_map, self.n_splits = policy_map, n_splits

    def split(self, X, y=None, groups=None):
        g = X.index.map(self.policy_map)
        return GroupKFold(n_splits=self.n_splits).split(X, y, g)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def policy_holdout(df, test_size=0.2, seed=SEED):
    """충전 정책 단위 Hold-out: 같은 정책 셀은 Train/Valid 한쪽에만"""
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                          .split(df, groups=df["policy"]))
    return df.iloc[tr_idx], df.iloc[va_idx]


# ---------------------------------------------------------------- 모델
def build_models(cv, feats):
    """후보 모델 4개. ElasticNet, LightGBM은 Train 안에서 GridSearch로 튜닝"""
    enet = GridSearchCV(
        make_pipeline(StandardScaler(), ElasticNet(max_iter=50_000, random_state=SEED)),
        {"elasticnet__alpha": np.logspace(-4, 0, 13),
         "elasticnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
        cv=cv, scoring="neg_mean_squared_error")

    lgbm = GridSearchCV(
        LGBMRegressor(random_state=SEED, verbose=-1, min_child_samples=3, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8),
        {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1],
         "num_leaves": [4, 8], "min_child_samples": [3, 5]},
        cv=cv, scoring="neg_mean_squared_error")

    return {
        "Baseline (Variance)": (make_pipeline(StandardScaler(), LinearRegression()), BASELINE_FEATURES),
        "ElasticNet": (enet, feats),
        "LightGBM": (lgbm, feats),
        "Voting (EN+LGBM)": ("voting", feats),   # 튜닝된 두 모델로 학습 시점에 구성
    }


def make_voting(fitted_enet, fitted_lgbm):
    return VotingRegressor([("enet", fitted_enet.best_estimator_),
                            ("lgbm", fitted_lgbm.best_estimator_)])


# ---------------------------------------------------------------- 평가
def metrics(y_log_true, y_log_pred):
    t, p = 10 ** np.asarray(y_log_true), 10 ** np.asarray(y_log_pred)
    return {"MAPE(%)": mean_absolute_percentage_error(t, p) * 100,
            "RMSE(cycle)": np.sqrt(mean_squared_error(t, p))}


def evaluate(name, model, feats, tr, va, tests, cv):
    Xtr, ytr = tr[feats], tr[TARGET]
    cv_pred = cross_val_predict(model, Xtr, ytr, cv=cv)        # Train CV (튜닝 포함 재학습)
    model.fit(Xtr, ytr)
    rows = {"Train (Batch1 CV)": metrics(ytr, cv_pred),
            "Valid (Batch1 Hold-out)": metrics(va[TARGET], model.predict(va[feats]))}
    preds = []
    for tname, df in tests.items():
        p = model.predict(df[feats])
        rows[tname] = metrics(df[TARGET], p)
        preds.append(pd.DataFrame({"model": name, "split": tname, "cycle_life": df["cycle_life"],
                                   "pred": 10 ** p}, index=df.index))
    preds.append(pd.DataFrame({"model": name, "split": "Valid", "cycle_life": va["cycle_life"],
                               "pred": 10 ** model.predict(va[feats])}, index=va.index))
    return rows, pd.concat(preds), model


def report(all_rows):
    """
    과제 리포팅 형식(Notion): split별 MAPE + Gap
    MAPE는 낮을수록 좋으므로, Notion의 "(+) = 문제 의심" 의미가 유지되도록
    Gap은 '나중 단계 오차 - 앞 단계 오차'로 계산 (비고에 계산식 명시)
    """
    out = []
    for name, r in all_rows.items():
        m = {k: v["MAPE(%)"] for k, v in r.items()}
        rm = {k: v["RMSE(cycle)"] for k, v in r.items()}
        tr, va = m["Train (Batch1 CV)"], m["Valid (Batch1 Hold-out)"]
        b2, b3 = m["Test (Batch2)"], m["Test (Batch3)"]
        out += [
            [name, "Train (Batch1 CV)", tr, rm["Train (Batch1 CV)"], "정책 단위 5-fold CV 평균"],
            [name, "Valid (Batch1 Hold-out)", va, rm["Valid (Batch1 Hold-out)"], "정책 단위 Hold-out 20%"],
            [name, "Test (Batch2)", b2, rm["Test (Batch2)"], "최종 평가"],
            [name, "Gap (Train-Valid)", va - tr, np.nan, "Valid−Train 오차, (+): 과적합 의심"],
            [name, "Gap (Valid-Test)", b2 - va, np.nan, "Test−Valid 오차, (+): 배치 간 일반화 저하 의심"],
            [name, "Gap (Target-Test)", b2 - TARGET_PAPER_MAPE, np.nan, "Test−9.1%, (+): 원 논문 목표 미달"],
            [name, "Test (Batch3)", b3, rm["Test (Batch3)"], "추가 검증"],
            [name, "Gap (Batch2-Batch3)", b2 - b3, np.nan, "Batch2−Batch3 오차, (+): Batch2가 더 나쁨"],
            [name, "Gap (Target-Test B3)", b3 - TARGET_PAPER_MAPE, np.nan, "Batch3−9.1%, (+): 원 논문 목표 미달"],
        ]
    return pd.DataFrame(out, columns=["model", "split", "MAPE(%)", "RMSE(cycle)", "note"])


# ---------------------------------------------------------------- 실행
def main():
    data = load_features()
    b1 = data["b1"]
    tr, va = policy_holdout(b1)
    assert not set(tr["policy"]) & set(va["policy"]), "정책 누수"
    tests = {"Test (Batch2)": data["b2"], "Test (Batch3)": data["b3"]}
    cv = PolicyGroupKFold(b1["policy"].to_dict(), n_splits=5)
    feats = select_features(tr)
    print("선택 피처 (Train 기준):", feats)
    print(f"Train {len(tr)}셀({tr['policy'].nunique()}정책) / Valid {len(va)}셀({va['policy'].nunique()}정책) "
          f"/ Test B2 {len(data['b2'])}셀, B3 {len(data['b3'])}셀")

    all_rows, all_preds, fitted = {}, [], {}
    for name, (model, mfeats) in build_models(cv, feats).items():
        if model == "voting":
            model = make_voting(fitted["ElasticNet"], fitted["LightGBM"])
        rows, preds, fitted[name] = evaluate(name, model, mfeats, tr, va, tests, cv)
        all_rows[name] = rows
        all_preds.append(preds)
        print(f"  {name:22s} " + " | ".join(f"{k.split(' (')[0]} {v['MAPE(%)']:.1f}%"
                                              for k, v in rows.items()))

    en = fitted["ElasticNet"]
    print("\nElasticNet best:", en.best_params_)
    coef = pd.Series(en.best_estimator_[-1].coef_, index=feats).sort_values(key=abs, ascending=False)
    print("ElasticNet 계수 (표준화 기준):\n", coef.round(4).to_string())
    print("LightGBM best:", fitted["LightGBM"].best_params_)

    RESULT_DIR.mkdir(exist_ok=True)
    perf = report(all_rows)
    perf.round(2).to_csv(RESULT_DIR / "model_performance_all.csv", index=False)
    final = perf[perf["model"] == FINAL_MODEL].drop(columns="model")
    final.round(2).to_csv(RESULT_DIR / "model_performance.csv", index=False)
    pd.concat(all_preds).round(1).to_csv(RESULT_DIR / "predictions.csv")
    print(f"\n=== 최종 모델: {FINAL_MODEL} ===")
    print(final.round(2).to_string(index=False))
    print("\n저장: results/model_performance.csv(최종), model_performance_all.csv(전체 후보), predictions.csv")
    return perf


if __name__ == "__main__":
    main()
