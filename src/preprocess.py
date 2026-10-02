"""
전처리 모듈: .mat 원본 → 정리된 사이클 요약 DataFrame (+ 피처용 Qdlin)

EDA(notebooks/01_EDA.ipynb)에서 확정한 규칙
1. 모든 배치: cycle_life가 NaN인 셀 제거 (특수 실험 VarCharge, SLOWCYCLE 등)
2. Batch1: EOL 도달 전 기록이 끝난 셀 10개 제거
   - A그룹 c00~c04: 실험 기간 내 미종료 (Batch2에 연장 데이터 없음)
   - B그룹 c08, c10, c12, c13, c22: EOL 전 기록 종료
3. 사이클 단위: QDischarge 0.5Ah 이하(측정 오류) 제거, 노이즈 스파이크는 rolling median으로 보정

사용 예 (프로젝트 루트에서):
    python -m src.preprocess          # 3개 배치 처리 후 data/processed/에 저장
"""
import gc
import logging
from pathlib import Path

import mat73
import numpy as np
import pandas as pd
import scipy.io as sio

logging.getLogger().setLevel(logging.CRITICAL)   # barcode 등 string 필드 경고 숨김

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",  # train
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",  # test
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",  # 추가 test
}
NOMINAL = 1.1            # 정격 용량 (Ah)
EOL = NOMINAL * 0.8      # 0.88 Ah
QD_MIN_VALID = 0.5       # 이 값 이하는 측정 오류로 간주

# EDA에서 찾은 Batch1 미완료 셀
B1_UNFINISHED = [f"b1_c{i:02d}" for i in [0, 1, 2, 3, 4, 8, 10, 12, 13, 22]]


# ---------------------------------------------------------------- 로딩
def load_mat(path):
    """MATLAB v7.3(HDF5)은 mat73, 그 외는 scipy로 읽기"""
    try:
        return mat73.loadmat(path)
    except Exception:
        return sio.loadmat(path, simplify_cells=True)


def to_list_of_dicts(batch):
    """mat73 결과 {key: [셀1, 셀2, ...]} → [{셀1}, {셀2}, ...]"""
    if isinstance(batch, dict):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        return [{k: batch[k][i] for k in keys} for i in range(n)]
    return batch


def load_batch(name):
    mat = load_mat(RAW_DIR / BATCH_FILES[name])
    return to_list_of_dicts(mat["batch"])


def _scalar(x):
    return float(np.ravel(x)[0]) if np.size(x) else np.nan


# ---------------------------------------------------------------- 추출
def extract_summary(batch, batch_name):
    """셀별 summary(사이클 단위 요약)를 하나의 DataFrame으로"""
    rows = []
    for i, cell in enumerate(batch):
        df = pd.DataFrame({k: np.ravel(v) for k, v in cell["summary"].items()})
        df["cell_id"] = f"{batch_name}_c{i:02d}"
        df["policy"] = cell.get("policy_readable")
        df["cycle_life"] = _scalar(cell["cycle_life"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def extract_qdlin(batch, batch_name, cycles=(10, 100)):
    """
    지정 사이클의 Qdlin(전압 격자로 보간한 방전용량 곡선)을 셀별로 추출.
    ΔQ(V) = Qdlin[100] - Qdlin[10] 피처를 만들 때 사용.
    반환: {cell_id: {cycle: np.ndarray}}

    주의(미확인): cycles 리스트의 인덱스가 사이클 번호와 1:1인지 배치별로 확인 필요.
    """
    out = {}
    for i, cell in enumerate(batch):
        qd = cell["cycles"]["Qdlin"]
        out[f"{batch_name}_c{i:02d}"] = {
            c: np.ravel(qd[c]) for c in cycles if c < len(qd)
        }
    return out


# ---------------------------------------------------------------- 정리
def clean_cells(df):
    """셀 단위 제거: cycle_life NaN + Batch1 미완료 셀"""
    before = df["cell_id"].nunique()
    df = df[df["cycle_life"].notna()]
    df = df[~df["cell_id"].isin(B1_UNFINISHED)]
    print(f"  셀 정리: {before} → {df['cell_id'].nunique()}")
    return df.copy()


def remove_noise(df, window=5, tol=0.01):
    """
    사이클 단위 정리
    - QDischarge 0.5Ah 이하 행 제거 (측정 오류)
    - 이동 중앙값과 tol(Ah) 이상 차이 나는 스파이크는 중앙값으로 대체
    """
    df = df[df["QDischarge"] > QD_MIN_VALID].sort_values(["cell_id", "cycle"]).copy()
    med = (df.groupby("cell_id")["QDischarge"]
             .transform(lambda s: s.rolling(window, center=True, min_periods=1).median()))
    spike = (df["QDischarge"] - med).abs() > tol
    df["QD_raw"] = df["QDischarge"]
    df.loc[spike, "QDischarge"] = med[spike]
    print(f"  스파이크 보정: {spike.sum()}행")
    return df


# ---------------------------------------------------------------- 실행
def process_batch(name, force=False):
    """원본 → 정리된 summary, Qdlin 저장. 캐시가 있으면 바로 읽음"""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    sum_path = PROC_DIR / f"{name}_summary_clean.pkl"
    qd_path = PROC_DIR / f"{name}_qdlin.pkl"
    if sum_path.exists() and qd_path.exists() and not force:
        return pd.read_pickle(sum_path), pd.read_pickle(qd_path)

    print(f"[{name}] 로딩 중...")
    batch = load_batch(name)
    df = extract_summary(batch, name)
    qdlin = extract_qdlin(batch, name)
    del batch
    gc.collect()

    df = remove_noise(clean_cells(df))
    qdlin = {k: v for k, v in qdlin.items() if k in set(df["cell_id"])}

    df.to_pickle(sum_path)
    pd.to_pickle(qdlin, qd_path)
    print(f"  저장: {sum_path.name}, {qd_path.name} ({df['cell_id'].nunique()}셀)")
    return df, qdlin


if __name__ == "__main__":
    for b in BATCH_FILES:
        process_batch(b)
