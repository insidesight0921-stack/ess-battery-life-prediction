"""
모델 진단 2가지
1) LightGBM 피처 중요도 vs ElasticNet 계수 → 두 모델이 같은 피처에 의존하는가?
2) 예측값 범위 → 트리 모델(LightGBM)은 Train 수명 범위 밖을 예측하지 못하는가? (외삽 가설)

train.py와 같은 분할(정책 단위 Hold-out)·같은 튜닝으로 학습
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import select_features
from src.train import (ROOT, TARGET, PolicyGroupKFold, build_models, load_features,
                       policy_holdout)

plt.rcParams.update({"font.family": ["AppleGothic", "Noto Sans CJK JP", "sans-serif"],
                     "axes.unicode_minus": False})
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

data = load_features()
b1 = data["b1"]
tr, va = policy_holdout(b1)
cv = PolicyGroupKFold(b1["policy"].to_dict(), n_splits=5)
FEATURES = select_features(tr)
print("선택 피처:", FEATURES)
models = build_models(cv, FEATURES)
enet, _ = models["ElasticNet"]
lgbm, _ = models["LightGBM"]
enet.fit(tr[FEATURES], tr[TARGET])
lgbm.fit(tr[FEATURES], tr[TARGET])

# ---------------------------------------------------------------- 1) 피처 의존도 비교
best_lgbm = lgbm.best_estimator_
imp = pd.DataFrame({
    "EN_coef(표준화)": enet.best_estimator_[-1].coef_,
    "LGBM_split(분기 횟수)": best_lgbm.booster_.feature_importance("split"),
    "LGBM_gain(오차 감소량)": best_lgbm.booster_.feature_importance("gain"),
}, index=FEATURES)
imp["LGBM_gain_%"] = imp["LGBM_gain(오차 감소량)"] / imp["LGBM_gain(오차 감소량)"].sum() * 100
print("=== 1) 피처 의존도 ===")
print(imp.sort_values("LGBM_gain_%", ascending=False).round(4).to_string())

# ---------------------------------------------------------------- 2) 예측값 범위 (외삽)
lo, hi = tr["cycle_life"].min(), tr["cycle_life"].max()
print(f"\n=== 2) 예측 범위 (Train 수명 범위 {lo:.0f} ~ {hi:.0f}) ===")
rows = []
preds = {}
for bname in ["b2", "b3"]:
    df = data[bname]
    for mname, m in [("ElasticNet", enet), ("LightGBM", lgbm)]:
        p = 10 ** m.predict(df[FEATURES])
        preds[(bname, mname)] = p
        rows.append([bname, mname, df["cycle_life"].min(), df["cycle_life"].max(), p.min(), p.max(),
                     (df["cycle_life"] > hi).sum(), (p > hi).sum(), (df["cycle_life"] < lo).sum(), (p < lo).sum()])
rng = pd.DataFrame(rows, columns=["batch", "model", "실제_min", "실제_max", "예측_min", "예측_max",
                                  "실제>Train최대(셀)", "예측>Train최대(셀)", "실제<Train최소(셀)", "예측<Train최소(셀)"])
print(rng.round(0).to_string(index=False))

# 범위 밖 셀만의 MAPE
print("\n=== Train 범위 밖 셀의 MAPE ===")
for bname in ["b2", "b3"]:
    y = data[bname]["cycle_life"].values
    out = (y > hi) | (y < lo)
    for mname in ["ElasticNet", "LightGBM"]:
        p = preds[(bname, mname)]
        e = np.abs(p - y) / y * 100
        print(f"  {bname} {mname:10s} 범위 밖 {out.sum():2d}셀 MAPE {e[out].mean():5.1f}% | 범위 안 {(~out).sum():2d}셀 MAPE {e[~out].mean():5.1f}%")

# 그림: 실제 vs 예측
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
for ax, bname in zip(axes, ["b2", "b3"]):
    y = data[bname]["cycle_life"].values
    for mname, c in [("ElasticNet", "#0f766e"), ("LightGBM", "#ea580c")]:
        ax.scatter(y, preds[(bname, mname)], s=35, alpha=.75, color=c, label=mname)
    ax.axhspan(lo, hi, color="gray", alpha=.12, label="Train 수명 범위")
    ax.plot([300, 2000], [300, 2000], "k:", lw=1, label="정답선 (예측 = 실제)")
    ax.set(title=f"{'Batch2 (Test)' if bname == 'b2' else 'Batch3 (추가 Test)'}: 실제 vs 예측",
           xlabel="실제 Cycle Life", ylabel="예측 Cycle Life", xlim=(300, 2000), ylim=(300, 2000))
    ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "diag_extrapolation.png", dpi=150)
imp.round(4).to_csv(ROOT / "results" / "diag_feature_dependence.csv")
rng.round(1).to_csv(ROOT / "results" / "diag_prediction_range.csv", index=False)
print("\n저장: results/figures/diag_extrapolation.png, results/diag_*.csv")
