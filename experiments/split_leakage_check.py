"""
분할 방식에 따른 Valid 성능 차이 확인 (무작위 vs 충전 정책 단위)
- 한 번의 분할은 운에 좌우되므로 seed 30개로 반복
- 모델은 튜닝 없이 고정 (분할 효과만 보기 위해)
"""
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.preprocess import PROC_DIR
from src.features import select_features
from src.train import BASELINE_FEATURES, TARGET

b1 = pd.read_csv(PROC_DIR / "features_b1.csv", index_col=0)
models = {"Baseline": (lambda: make_pipeline(StandardScaler(), LinearRegression()), BASELINE_FEATURES),
          "ElasticNet": (lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=50000)), None)}
rows = []
for seed in range(30):
    splits = {"random": train_test_split(b1, test_size=0.2, random_state=seed)}
    tri, vai = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(b1, groups=b1.policy))
    splits["policy"] = (b1.iloc[tri], b1.iloc[vai])
    for sname, (tr, va) in splits.items():
        leak = va.policy.isin(tr.policy).mean()
        for mname, (mk, feats) in models.items():
            feats = feats or select_features(tr)
            m = mk().fit(tr[feats], tr[TARGET])
            rows.append([seed, sname, mname, leak, mape(10**va[TARGET], 10**m.predict(va[feats])) * 100])
df = pd.DataFrame(rows, columns=["seed", "split", "model", "valid_leak_ratio", "valid_mape"])
print(df.groupby(["model", "split"]).agg(mape_mean=("valid_mape", "mean"), mape_std=("valid_mape", "std"),
                                         leak=("valid_leak_ratio", "mean")).round(2))
df.to_csv("results/split_leakage_check.csv", index=False)
