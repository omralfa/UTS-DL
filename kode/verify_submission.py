import os
import numpy as np
import pandas as pd
from utils import DATA_DIR, PROJECT_DIR, CLASSES

def verify():
    print("=" * 70)
    print("VERIFYING SUBMISSION PREDICTION FILES")
    print("=" * 70)

    # 1. prediksi_klasifikasi.csv
    p1 = pd.read_csv(os.path.join(PROJECT_DIR, "prediksi_klasifikasi.csv"))
    te1 = pd.read_csv(os.path.join(DATA_DIR, "klasifikasi_test_meta.csv"))
    assert list(p1.columns) == ["id_unit", "kelas"], f"Bad columns: {p1.columns}"
    assert len(p1) == len(te1), f"Row count mismatch: {len(p1)} vs {len(te1)}"
    assert (p1["id_unit"].values == te1["id_unit"].values).all(), "id_unit order/set mismatch!"
    assert not p1.isna().any().any(), "NaNs found in prediksi_klasifikasi.csv!"
    assert set(p1["kelas"].unique()).issubset(set(CLASSES)), f"Invalid classes: {set(p1['kelas'].unique()) - set(CLASSES)}"
    print(f"[OK] prediksi_klasifikasi.csv: {len(p1)} rows, classes={p1['kelas'].value_counts().to_dict()}")

    # 2. prediksi_rul.csv
    p2 = pd.read_csv(os.path.join(PROJECT_DIR, "prediksi_rul.csv"))
    te2 = pd.read_csv(os.path.join(DATA_DIR, "rul_test.csv.gz"))
    te2_units = np.sort(te2["id_unit"].unique())
    assert list(p2.columns) == ["id_unit", "rul"], f"Bad columns: {p2.columns}"
    assert len(p2) == len(te2_units), f"Row count mismatch: {len(p2)} vs {len(te2_units)}"
    assert set(p2["id_unit"].values) == set(te2_units), "id_unit mismatch in prediksi_rul.csv!"
    assert not p2.isna().any().any(), "NaNs found in prediksi_rul.csv!"
    assert (p2["rul"] >= 0).all() and (p2["rul"] <= 150).all(), "RUL out of [0, 150]!"
    print(f"[OK] prediksi_rul.csv: {len(p2)} rows, min={p2['rul'].min():.2f}, med={p2['rul'].median():.2f}, max={p2['rul'].max():.2f}")

    # 3. cluster.csv
    p3 = pd.read_csv(os.path.join(PROJECT_DIR, "cluster.csv"))
    te3 = pd.read_csv(os.path.join(DATA_DIR, "clustering_meta.csv"))
    assert list(p3.columns) == ["id_sampel", "cluster"], f"Bad columns: {p3.columns}"
    assert len(p3) == len(te3), f"Row count mismatch: {len(p3)} vs {len(te3)}"
    assert (p3["id_sampel"].values == te3["id_sampel"].values).all(), "id_sampel order/set mismatch!"
    assert not p3.isna().any().any(), "NaNs found in cluster.csv!"
    assert np.issubdtype(p3["cluster"].dtype, np.integer), f"cluster dtype not integer: {p3['cluster'].dtype}"
    print(f"[OK] cluster.csv: {len(p3)} rows, K={p3['cluster'].nunique()}, counts={p3['cluster'].value_counts().sort_index().to_dict()}")
    print("\nALL 3 PREDICTION FILES PASSED STRICT VERIFICATION!")


if __name__ == "__main__":
    verify()
