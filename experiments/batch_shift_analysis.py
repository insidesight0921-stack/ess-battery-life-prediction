"""
Batch2 Test MAPE가 30%인 원인 분석

가설
 H1. 외삽 문제: Batch2 단수명 셀의 피처가 Train 범위 밖이라 못 맞힌다
 H2. 관계 변화: 같은 피처 값이어도 배치(실험 조건)에 따라 수명이 다르다
확인
 1) 같은 충전 정책인데 배치가 다른 셀의 수명 비교 → 배치 효과 크기
 2) dQ_log_var vs log(수명) 관계를 그룹별로 겹쳐 보기 + Train 회귀선 대비 잔차
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocess import PROC_DIR, ROOT

plt.rcParams.update({"font.family": ["AppleGothic", "Noto Sans CJK JP", "sans-serif"],
                     "axes.unicode_minus": False, "font.size": 12})
FIG = ROOT / "results" / "figures"

f = pd.concat([pd.read_csv(PROC_DIR / f"features_{b}.csv", index_col=0) for b in ["b1", "b2", "b3"]])
f["newstructure"] = f["policy"].str.contains("newstructure")
f["policy_base"] = f["policy"].str.replace("-newstructure", "", regex=False)
f["group"] = np.select(
    [f.batch == "b1", (f.batch == "b2") & ~f.newstructure, (f.batch == "b2") & f.newstructure, f.batch == "b3"],
    ["B1 (Train)", "B2 일반", "B2 newstructure", "B3 (newstructure)"], default="기타")
GROUPS = ["B1 (Train)", "B2 일반", "B2 newstructure", "B3 (newstructure)"]
COLORS = dict(zip(GROUPS, ["#0f766e", "#dc2626", "#f59e0b", "#6366f1"]))

# ---------------------------------------------------------------- 1) 같은 정책, 다른 배치
same = (f.groupby(["policy_base", "group"])["cycle_life"].agg(["size", "mean"]).reset_index()
          .groupby("policy_base").filter(lambda d: d["group"].nunique() > 1))
print("=== 1) 같은 충전 정책, 다른 그룹의 평균 수명 ===")
print(same.round(0).to_string(index=False))

# ---------------------------------------------------------------- 2) 피처-수명 관계
tr = f[f.group == "B1 (Train)"]
coef = np.polyfit(tr["dQ_log_var"], tr["log_cycle_life"], 1)
f["pred_b1line"] = 10 ** np.polyval(coef, f["dQ_log_var"])
f["resid_%"] = (f["pred_b1line"] - f["cycle_life"]) / f["cycle_life"] * 100   # (+) = 길게 예측
lo, hi = tr["dQ_log_var"].min(), tr["dQ_log_var"].max()
summary = f.groupby("group").agg(
    cells=("cycle_life", "size"), life_median=("cycle_life", "median"),
    dQ_var_median=("dQ_log_var", "median"),
    in_train_range=("dQ_log_var", lambda s: s.between(lo, hi).mean()),
    resid_mean_pct=("resid_%", "mean"), mape_pct=("resid_%", lambda s: s.abs().mean())).reindex(GROUPS)
print(f"\n=== 2) Train 회귀선(dQ_log_var 1개) 기준 그룹별 잔차, Train dQ_log_var 범위 {lo:.2f}~{hi:.2f} ===")
print(summary.round(2).to_string())

# 그림
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
for g in GROUPS:
    d = f[f.group == g]
    ax.scatter(d["dQ_log_var"], d["cycle_life"], s=40, alpha=.8, color=COLORS[g], label=f"{g} (n={len(d)})")
xs = np.linspace(f["dQ_log_var"].min(), f["dQ_log_var"].max(), 50)
ax.plot(xs, 10 ** np.polyval(coef, xs), "k--", lw=1.5, label="Train 회귀선")
ax.axvspan(lo, hi, color="gray", alpha=.1)
ax.set(yscale="log", xlabel="dQ_log_var (ΔQ 분산의 log)", ylabel="Cycle Life (log 축)",
       title="피처-수명 관계: 그룹마다 선이 다르다")
ax.legend(fontsize=9)

ax = axes[1]
pols = same["policy_base"].unique()
w = 0.2
for i, g in enumerate(GROUPS):
    vals = [same[(same.policy_base == p) & (same.group == g)]["mean"].values for p in pols]
    ax.bar(np.arange(len(pols)) + (i - 1.5) * w, [v[0] if len(v) else 0 for v in vals], w,
           color=COLORS[g], label=g)
ax.set_xticks(np.arange(len(pols)))
ax.set_xticklabels(pols, rotation=15)
ax.set(ylabel="평균 Cycle Life", title="같은 충전 정책인데 그룹마다 수명이 다르다")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "batch_shift.png", dpi=150)
summary.round(3).to_csv(ROOT / "results" / "batch_shift_summary.csv")
same.round(1).to_csv(ROOT / "results" / "batch_shift_same_policy.csv", index=False)
print("\n저장: results/figures/batch_shift.png, results/batch_shift_*.csv")
