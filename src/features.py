"""
피처 생성 모듈: 정리된 사이클 데이터 → 셀 1개 = 1행 피처 테이블

입력: data/processed/{b}_summary_clean.pkl, {b}_qdlin.pkl  (src/preprocess.py 결과)
출력: data/processed/features_{b}.csv

피처 설계 근거
- EDA: 초기 100사이클의 용량 숫자(QDischarge)는 셀 간 차이가 매우 작음
  → 방전 곡선 전체의 변화 ΔQ(V) = Qdlin[100] - Qdlin[10] 사용 (Severson et al., 2019)
- EDA: 충전시간이 짧을수록(급속충전) 수명이 짧음 → chargetime 포함
- 모든 피처는 사이클 100 이하 데이터만 사용 (미래 정보 누수 방지)

사용 예 (프로젝트 루트에서):
    python -m src.features
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.preprocess import BATCH_FILES, PROC_DIR

EARLY_CYCLE = 100   # 회귀 과제: 처음 100사이클만 사용
EPS = 1e-12         # log(0) 방지


def _log_abs(x):
    return np.log10(np.abs(x) + EPS)


def _value_at(g, col, cycle):
    """해당 사이클 값. 없으면(노이즈 제거로 빠진 경우) 가장 가까운 사이클 값"""
    idx = (g["cycle"] - cycle).abs().idxmin()
    return g.loc[idx, col]


def delta_q_features(qdlin_cell, c_early=10, c_late=100):
    """ΔQ(V) = Q_late(V) - Q_early(V) 곡선의 통계량"""
    dq = qdlin_cell[c_late] - qdlin_cell[c_early]
    return {
        "dQ_log_var": _log_abs(np.var(dq)),      # 논문의 핵심 단일 피처
        "dQ_log_min": _log_abs(np.min(dq)),
        "dQ_log_mean": _log_abs(np.mean(dq)),
        "dQ_skew": skew(dq),
        "dQ_kurt": kurtosis(dq),
        "dQ_at_2V": dq[-1],                       # 전압 격자 마지막 점(2.0V 부근, 미확인)
    }


def summary_features(g):
    """summary(사이클 요약) 기반 피처. g = 한 셀의 사이클 100 이하 데이터"""
    g = g.sort_values("cycle")
    qd2 = _value_at(g, "QDischarge", 2)
    qd100 = _value_at(g, "QDischarge", EARLY_CYCLE)

    tail = g[g["cycle"].between(91, EARLY_CYCLE)]           # 91~100 사이클 기울기
    slope, intercept = (np.polyfit(tail["cycle"], tail["QDischarge"], 1)
                        if len(tail) >= 2 else (np.nan, np.nan))

    return {
        "QD_c2": qd2,
        "QD_max_minus_c2": g["QDischarge"].max() - qd2,
        "QD_c100_minus_c2": qd100 - qd2,
        "QD_slope_91_100": slope,
        "QD_intercept_91_100": intercept,
        "IR_c2": _value_at(g, "IR", 2),
        "IR_c100_minus_c2": _value_at(g, "IR", EARLY_CYCLE) - _value_at(g, "IR", 2),
        "Tmax_mean": g["Tmax"].mean(),
        "Tavg_mean": g["Tavg"].mean(),
        "chargetime_mean_5": g[g["cycle"] <= 5]["chargetime"].mean(),
    }


def build_features(name):
    df = pd.read_pickle(PROC_DIR / f"{name}_summary_clean.pkl")
    qdlin = pd.read_pickle(PROC_DIR / f"{name}_qdlin.pkl")
    early = df[df["cycle"] <= EARLY_CYCLE]

    rows = []
    for cid, g in early.groupby("cell_id"):
        row = {"cell_id": cid, "batch": name, "policy": g["policy"].iloc[0]}
        row.update(summary_features(g))
        row.update(delta_q_features(qdlin[cid]))
        life = df.loc[df["cell_id"] == cid, "cycle_life"].iloc[0]
        row["cycle_life"] = life                       # 타깃 (회귀)
        row["log_cycle_life"] = np.log10(life)         # 타깃 변환
        row["label_550"] = int(life >= 550)            # 참고용 (분류)
        rows.append(row)

    feat = pd.DataFrame(rows).set_index("cell_id")
    out = PROC_DIR / f"features_{name}.csv"
    feat.to_csv(out)
    print(f"[{name}] {feat.shape[0]}셀 × {feat.shape[1]}열 → {out.name}")
    return feat


if __name__ == "__main__":
    for b in BATCH_FILES:
        build_features(b)
