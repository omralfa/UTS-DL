import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, WEIGHTS_DIR, PROJECT_DIR,
    ensure_dirs, set_seed, save_json, rmse, asymmetric_rul_score
)

ensure_dirs()

INFORMATIVE_SENSORS = [
    "s02", "s03", "s04", "s07", "s08", "s09", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]
UNINFORMATIVE_SENSORS = ["s01", "s05", "s06", "s10", "s16", "s18", "s19"]
WINDOW_LEN = 50


# ============================================================================
# 1. Leak-Free Preprocessing (Operating Regimes, Despiking, Causal Imputation)
# ============================================================================
class RULPreprocessor:
    def __init__(self, n_regimes=6, spike_z=5.0, ewma_alpha=0.15):
        self.n_regimes = n_regimes
        self.spike_z = spike_z
        self.ewma_alpha = ewma_alpha
        self.setel_mean = None
        self.setel_std = None
        self.km = None
        self.med_table = None
        self.mad_table = None

    def fit(self, df_train: pd.DataFrame):
        setel = df_train[["setel_1", "setel_2", "setel_3"]].values
        self.setel_mean = setel.mean(axis=0)
        self.setel_std = setel.std(axis=0) + 1e-8
        setel_norm = (setel - self.setel_mean) / self.setel_std
        self.km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10).fit(setel_norm)
        regimes = self.km.labels_

        tmp = df_train[INFORMATIVE_SENSORS].copy()
        tmp["regime"] = regimes
        self.med_table = tmp.groupby("regime")[INFORMATIVE_SENSORS].median()
        self.mad_table = tmp.groupby("regime")[INFORMATIVE_SENSORS].apply(
            lambda g: np.nanmedian(np.abs(g - np.nanmedian(g, axis=0)), axis=0)
        )
        # Convert mad_table to DataFrame aligned with med_table
        mad_arr = np.vstack(self.mad_table.values)
        self.mad_df = pd.DataFrame(
            np.maximum(1.4826 * mad_arr, 1e-4),
            index=self.med_table.index,
            columns=INFORMATIVE_SENSORS,
        )
        return self

    def transform(self, df: pd.DataFrame):
        out = df[["id_unit", "siklus"]].copy()
        setel = df[["setel_1", "setel_2", "setel_3"]].values
        setel_norm = (setel - self.setel_mean) / self.setel_std
        regimes = self.km.predict(setel_norm)
        out["regime"] = regimes

        # Within-regime robust normalization + spike masking
        X_raw = df[INFORMATIVE_SENSORS].values.astype(np.float32)
        meds = self.med_table.loc[regimes].values.astype(np.float32)
        mads = self.mad_df.loc[regimes].values.astype(np.float32)
        Z = (X_raw - meds) / mads
        Z[np.abs(Z) > self.spike_z] = np.nan

        # Strictly causal imputation (ffill within each unit, then 0.0) + causal EWMA
        z_df = pd.DataFrame(Z, columns=INFORMATIVE_SENSORS, index=df.index)
        z_df["id_unit"] = df["id_unit"].values
        z_imp = z_df.groupby("id_unit", sort=False)[INFORMATIVE_SENSORS].ffill().fillna(0.0)
        z_imp = z_imp.clip(-5.0, 5.0)

        ewma_cols = [f"{c}_ewm" for c in INFORMATIVE_SENSORS]
        z_ewm = z_imp.groupby(df["id_unit"].values, sort=False).transform(
            lambda s: s.ewm(alpha=self.ewma_alpha, adjust=False).mean()
        )
        z_ewm.columns = ewma_cols

        # Regime one-hot + normalized cycle index
        reg_oh = np.eye(self.n_regimes, dtype=np.float32)[regimes]
        cyc_feat = (df["siklus"].values.astype(np.float32) / 300.0)[:, None]

        feats = np.hstack([
            z_imp.values.astype(np.float32),
            z_ewm.values.astype(np.float32),
            cyc_feat,
            reg_oh,
        ])
        out_feats = feats  # (N_rows, 35)
        return out, out_feats


def build_unit_sequences(df_meta: pd.DataFrame, feats: np.ndarray, rul: np.ndarray = None,
                         window_len: int = WINDOW_LEN, stride: int = 1, last_only: bool = False):
    """Build strictly causal sliding windows per machine."""
    X_list, y_list, unit_list, cyc_list = [], [], [], []
    # Group indices by id_unit preserving cycle order
    df_tmp = df_meta.copy()
    df_tmp["_pos"] = np.arange(len(df_meta))
    for uid, grp in df_tmp.groupby("id_unit", sort=True):
        grp = grp.sort_values("siklus")
        pos = grp["_pos"].values
        F_u = feats[pos]  # (T, D)
        T = len(pos)
        if last_only:
            end_indices = [T - 1]
        else:
            # Start from cycle 10 onward (or all cycles) with left-padding when t < window_len
            end_indices = list(range(window_len - 1, T, stride))
            if (T - 1) not in end_indices:
                end_indices.append(T - 1)
            # Also add a few shorter early-life windows (left-padded) so model handles short test units
            for t in range(15, min(window_len - 1, T), max(stride * 2, 4)):
                end_indices.append(t)

        for t in end_indices:
            start = t - window_len + 1
            if start < 0:
                pad = np.repeat(F_u[0:1], -start, axis=0)
                win = np.vstack([pad, F_u[:t + 1]])
            else:
                win = F_u[start:t + 1]
            X_list.append(win)
            unit_list.append(uid)
            cyc_list.append(int(grp["siklus"].iloc[t]))
            if rul is not None:
                y_list.append(float(rul[pos[t]]))

    X_arr = np.asarray(X_list, dtype=np.float32)
    if rul is not None:
        return X_arr, np.asarray(y_list, dtype=np.float32), np.asarray(unit_list), np.asarray(cyc_list)
    return X_arr, np.asarray(unit_list), np.asarray(cyc_list)


def extract_tabular_features(X_seq: np.ndarray):
    """Summary features from causal window (B, W, 35) for LightGBM baseline."""
    # First 14 are z_imp, next 14 are z_ewm, index 28 is cycle, 29..34 regime
    last = X_seq[:, -1, :29]
    mean_50 = X_seq[:, :, :14].mean(axis=1)
    std_50 = X_seq[:, :, :14].std(axis=1)
    mean_15 = X_seq[:, -15:, :14].mean(axis=1)
    slope = (X_seq[:, -10:, 14:28].mean(axis=1) - X_seq[:, :10, 14:28].mean(axis=1))
    return np.hstack([last, mean_50, std_50, mean_15, slope])


class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return torch.from_numpy(self.X[idx])
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================================
# 2. Deep Learning Sequence Architectures
# ============================================================================
class BiGRUAttentionNet(nn.Module):
    def __init__(self, in_dim=35, hidden=96, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.in_proj(x)
        out, _ = self.gru(h)  # (B, W, 2H)
        w = F.softmax(self.attn(out), dim=1)
        ctx = (out * w).sum(dim=1)
        last = out[:, -1, :]
        pred = self.head(torch.cat([ctx, last], dim=1)).squeeze(-1)
        return pred


class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation, dropout=0.2):
        super().__init__()
        pad = dilation * 2  # kernel_size=3, causal-like or symmetric padding
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.conv1(x))))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.gelu(h + self.skip(x))


class DilatedTCNNet(nn.Module):
    def __init__(self, in_dim=35, channels=96, dropout=0.2):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, channels, 1)
        self.b1 = TCNBlock(channels, channels, dilation=1, dropout=dropout)
        self.b2 = TCNBlock(channels, channels, dilation=2, dropout=dropout)
        self.b3 = TCNBlock(channels, channels, dilation=4, dropout=dropout)
        self.b4 = TCNBlock(channels, channels, dilation=8, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        h = self.b4(self.b3(self.b2(self.b1(h))))
        feat = torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1)
        return self.head(feat).squeeze(-1)


class HybridCNNGRUTransformer(nn.Module):
    def __init__(self, in_dim=35, d_model=96, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout, activation="gelu", batch_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, 48), nn.Tanh(), nn.Linear(48, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        h = self.tr(h)
        w = F.softmax(self.attn_pool(h), dim=1)
        ctx = (h * w).sum(dim=1)
        last = h[:, -1, :]
        return self.head(torch.cat([ctx, last], dim=1)).squeeze(-1)


# ============================================================================
# 3. Training & MC-Dropout Uncertainty
# ============================================================================
def rul_loss(pred, target, asym_weight=0.15):
    """Smooth Huber/MSE loss + mild asymmetric penalty for late (over-optimistic) predictions."""
    diff = pred - target
    huber = F.smooth_l1_loss(pred, target, beta=10.0)
    # Mild extra penalty when pred > target (over-optimistic)
    late_pen = torch.mean(F.relu(diff) ** 2) * 0.002
    return huber + asym_weight * late_pen


def train_seq_model(arch, X_tr, y_tr, X_va, y_va, device, epochs=18, batch_size=512, lr=2e-3, seed=42):
    set_seed(seed)
    if arch == "bigru":
        model = BiGRUAttentionNet(in_dim=X_tr.shape[2], hidden=64).to(device)
    elif arch == "tcn":
        model = DilatedTCNNet(in_dim=X_tr.shape[2], channels=64).to(device)
    elif arch == "hybrid":
        model = HybridCNNGRUTransformer(in_dim=X_tr.shape[2], d_model=64).to(device)
    else:
        raise ValueError(arch)

    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=512, shuffle=False, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_rmse = 1e9
    best_state = None
    best_pred = None
    hist = {"train_loss": [], "val_rmse": []}

    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = rul_loss(pred, yb, asym_weight=0.10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()

        vp = predict_seq(model, va_loader, device)
        vr = rmse(y_va, np.clip(vp, 0, 150))
        hist["train_loss"].append(round(tot / n, 4))
        hist["val_rmse"].append(round(vr, 4))
        if vr < best_rmse:
            best_rmse = vr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_pred = np.clip(vp, 0, 150)

    model.load_state_dict(best_state)
    return model, best_rmse, best_pred, hist


@torch.no_grad()
def predict_seq(model, loader, device, mc_dropout=False):
    if mc_dropout:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    else:
        model.eval()
    preds = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def range_breakdown(y_true, y_pred):
    out = {}
    ranges = [("0-30", 0, 30), ("31-80", 30.001, 80), ("81-150", 80.001, 150.1)]
    for name, lo, hi in ranges:
        m = (y_true >= lo) & (y_true <= hi)
        if m.any():
            out[name] = {
                "count": int(m.sum()),
                "rmse": round(rmse(y_true[m], y_pred[m]), 3),
                "asym_score_mean": round(asymmetric_rul_score(y_true[m], y_pred[m]) / m.sum(), 3),
                "asym_score_sum": round(asymmetric_rul_score(y_true[m], y_pred[m]), 1),
            }
    return out


# ============================================================================
# 4. Main Task 2 Pipeline
# ============================================================================
def run_task2(epochs=18):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 2] Device: {device}")

    rul_tr = pd.read_csv(os.path.join(DATA_DIR, "rul_train.csv.gz"))
    rul_te = pd.read_csv(os.path.join(DATA_DIR, "rul_test.csv.gz"))

    # Compute capped RUL labels (cap = 150)
    max_cyc = rul_tr.groupby("id_unit")["siklus"].transform("max")
    rul_tr["rul"] = (max_cyc - rul_tr["siklus"]).clip(upper=150).astype(np.float32)

    # Split STRICTLY by machine (id_unit): 720 train, 180 val
    units = np.sort(rul_tr["id_unit"].unique())
    rng = np.random.RandomState(42)
    perm = rng.permutation(units)
    tr_units = set(perm[:720])
    va_units = set(perm[720:])

    df_tr = rul_tr[rul_tr["id_unit"].isin(tr_units)].copy().reset_index(drop=True)
    df_va = rul_tr[rul_tr["id_unit"].isin(va_units)].copy().reset_index(drop=True)

    # Fit preprocessor on training machines ONLY
    prep = RULPreprocessor(n_regimes=6, spike_z=5.0, ewma_alpha=0.15).fit(df_tr)
    meta_tr, F_tr = prep.transform(df_tr)
    meta_va, F_va = prep.transform(df_va)
    meta_te, F_te = prep.transform(rul_te)

    # Build sliding windows (stride=2 on train for speed & diversity, stride=2 on val all-cycles)
    X_tr, y_tr, _, _ = build_unit_sequences(meta_tr, F_tr, df_tr["rul"].values, window_len=WINDOW_LEN, stride=2)
    X_va_all, y_va_all, u_va_all, c_va_all = build_unit_sequences(meta_va, F_va, df_va["rul"].values, window_len=WINDOW_LEN, stride=2)
    X_te, te_units, _ = build_unit_sequences(meta_te, F_te, None, window_len=WINDOW_LEN, last_only=True)

    # Also build realistic test-like validation set: 1 random cutoff >= cycle 50 per val machine (180 machines)
    va_cutoff_idx = []
    for u in sorted(va_units):
        idxs = np.where((u_va_all == u) & (c_va_all >= 50))[0]
        va_cutoff_idx.append(rng.choice(idxs))
    va_cutoff_idx = np.array(va_cutoff_idx)
    X_va_cut, y_va_cut = X_va_all[va_cutoff_idx], y_va_all[va_cutoff_idx]

    print(f"[Task 2] Windows: train={X_tr.shape}, val_all={X_va_all.shape}, val_cut(180 machines)={X_va_cut.shape}, test={X_te.shape}")

    results = {
        "n_train_units": len(tr_units),
        "n_val_units": len(va_units),
        "n_test_units": len(te_units),
        "informative_sensors": INFORMATIVE_SENSORS,
        "dropped_uninformative_sensors": UNINFORMATIVE_SENSORS,
        "window_len": WINDOW_LEN,
        "models": {},
    }

    # 1. Classical non-DL Baseline: LightGBM on causal window summary features
    print("[Task 2] Training non-DL baseline: LightGBM...")
    Xtr_tab = extract_tabular_features(X_tr)
    Xva_tab = extract_tabular_features(X_va_all)
    Xvc_tab = extract_tabular_features(X_va_cut)
    gbm = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    gbm.fit(Xtr_tab, y_tr, eval_set=[(Xva_tab, y_va_all)], callbacks=[lgb.early_stopping(30, verbose=False)])
    p_lgb_all = np.clip(gbm.predict(Xva_tab), 0, 150)
    p_lgb_cut = np.clip(gbm.predict(Xvc_tab), 0, 150)
    results["models"]["lightgbm_baseline"] = {
        "val_all_rmse": round(rmse(y_va_all, p_lgb_all), 3),
        "val_all_asym_mean": round(asymmetric_rul_score(y_va_all, p_lgb_all) / len(y_va_all), 3),
        "val_180cut_rmse": round(rmse(y_va_cut, p_lgb_cut), 3),
        "val_180cut_asym_sum": round(asymmetric_rul_score(y_va_cut, p_lgb_cut), 1),
        "range_breakdown": range_breakdown(y_va_all, p_lgb_all),
    }
    print(f"  LightGBM -> all_RMSE={results['models']['lightgbm_baseline']['val_all_rmse']:.3f}, "
          f"180cut_RMSE={results['models']['lightgbm_baseline']['val_180cut_rmse']:.3f}, "
          f"180cut_Asym={results['models']['lightgbm_baseline']['val_180cut_asym_sum']:.1f}")

    # 2. Compare 3 Deep Learning Sequence Architectures
    va_cut_loader = DataLoader(SeqDataset(X_va_cut), batch_size=512, shuffle=False)
    va_all_loader = DataLoader(SeqDataset(X_va_all), batch_size=512, shuffle=False)
    te_loader = DataLoader(SeqDataset(X_te), batch_size=512, shuffle=False)

    arch_preds_cut = {}
    arch_hists = {}
    for arch in ["tcn", "bigru", "hybrid"]:
        print(f"[Task 2] Training DL architecture: {arch}...")
        model, br, vp_all, hist = train_seq_model(arch, X_tr, y_tr, X_va_all, y_va_all, device, epochs=epochs, seed=42)
        vp_cut = np.clip(predict_seq(model, va_cut_loader, device), 0, 150)
        arch_preds_cut[arch] = vp_cut
        arch_hists[arch] = hist
        results["models"][arch] = {
            "val_all_rmse": round(br, 3),
            "val_all_asym_mean": round(asymmetric_rul_score(y_va_all, vp_all) / len(y_va_all), 3),
            "val_180cut_rmse": round(rmse(y_va_cut, vp_cut), 3),
            "val_180cut_asym_sum": round(asymmetric_rul_score(y_va_cut, vp_cut), 1),
            "range_breakdown": range_breakdown(y_va_all, vp_all),
        }
        print(f"  {arch:8s} -> all_RMSE={br:.3f}, 180cut_RMSE={rmse(y_va_cut, vp_cut):.3f}, "
              f"180cut_Asym={asymmetric_rul_score(y_va_cut, vp_cut):.1f}")

    # 3. Final Deep Ensemble (Hybrid + BiGRU across 3 seeds) + MC Dropout Uncertainty
    print("[Task 2] Training final 3-seed ensemble (Hybrid + BiGRU) for uncertainty & test predictions...")
    ens_va_all_samples = []
    ens_va_cut_samples = []
    ens_te_samples = []

    ensemble_configs = [("hybrid", 42), ("hybrid", 102), ("bigru", 202)]
    for arch, seed in ensemble_configs:
        model, br, _, _ = train_seq_model(arch, X_tr, y_tr, X_va_all, y_va_all, device, epochs=epochs + 4, seed=seed)
        if arch == "hybrid" and seed == 42:
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "task2_hybrid_seed42.pt"))
        # 15 MC Dropout passes per ensemble member -> 45 stochastic samples total!
        for _ in range(15):
            ens_va_all_samples.append(np.clip(predict_seq(model, va_all_loader, device, mc_dropout=True), 0, 150))
            ens_va_cut_samples.append(np.clip(predict_seq(model, va_cut_loader, device, mc_dropout=True), 0, 150))
            ens_te_samples.append(np.clip(predict_seq(model, te_loader, device, mc_dropout=True), 0, 150))

    ens_va_all_samples = np.asarray(ens_va_all_samples)  # (45, N_va_all)
    ens_va_cut_samples = np.asarray(ens_va_cut_samples)  # (45, 180)
    ens_te_samples = np.asarray(ens_te_samples)          # (45, 300)

    va_all_mu = ens_va_all_samples.mean(axis=0)
    va_all_sd = ens_va_all_samples.std(axis=0) + 1e-4
    va_cut_mu = ens_va_cut_samples.mean(axis=0)
    va_cut_sd = ens_va_cut_samples.std(axis=0) + 1e-4
    te_mu = ens_te_samples.mean(axis=0)
    te_sd = ens_te_samples.std(axis=0) + 1e-4

    # Conformal calibration of 90% prediction interval using validation normalized residuals
    norm_res = np.abs(y_va_all - va_all_mu) / va_all_sd
    q90 = float(np.quantile(norm_res, 0.90))
    lo90_cut = np.clip(va_cut_mu - q90 * va_cut_sd, 0, 150)
    hi90_cut = np.clip(va_cut_mu + q90 * va_cut_sd, 0, 150)
    cov90_cut = float(np.mean((y_va_cut >= lo90_cut) & (y_va_cut <= hi90_cut)))
    avg_width90 = float(np.mean(hi90_cut - lo90_cut))

    # Optimize mild conservative shift (alpha * sigma) for asymmetric score vs RMSE tradeoff
    best_alpha = 0.0
    best_comb = 1e18
    alpha_table = []
    for alpha in [0.0, 0.15, 0.25, 0.35, 0.50]:
        p_adj_all = np.clip(va_all_mu - alpha * va_all_sd, 0, 150)
        p_adj_cut = np.clip(va_cut_mu - alpha * va_cut_sd, 0, 150)
        r_all = rmse(y_va_all, p_adj_all)
        a_all = asymmetric_rul_score(y_va_all, p_adj_all) / len(y_va_all)
        r_cut = rmse(y_va_cut, p_adj_cut)
        a_cut = asymmetric_rul_score(y_va_cut, p_adj_cut)
        alpha_table.append({"alpha": alpha, "val_all_rmse": round(r_all, 3), "val_all_asym_mean": round(a_all, 3),
                            "val_180cut_rmse": round(r_cut, 3), "val_180cut_asym_sum": round(a_cut, 1)})
        # Pick alpha that reduces asymmetric score substantially while keeping RMSE within +0.25 of unshifted
        if r_all <= rmse(y_va_all, va_all_mu) + 0.25 and a_all < best_comb:
            best_comb = a_all
            best_alpha = alpha

    final_va_all = np.clip(va_all_mu - best_alpha * va_all_sd, 0, 150)
    final_va_cut = np.clip(va_cut_mu - best_alpha * va_cut_sd, 0, 150)
    final_te_pred = np.clip(te_mu - best_alpha * te_sd, 0, 150)

    results["final_ensemble"] = {
        "configs": ensemble_configs,
        "mc_passes_per_member": 15,
        "unshifted_val_all_rmse": round(rmse(y_va_all, va_all_mu), 3),
        "unshifted_val_all_asym_mean": round(asymmetric_rul_score(y_va_all, va_all_mu) / len(y_va_all), 3),
        "unshifted_val_180cut_rmse": round(rmse(y_va_cut, va_cut_mu), 3),
        "unshifted_val_180cut_asym_sum": round(asymmetric_rul_score(y_va_cut, va_cut_mu), 1),
        "chosen_alpha": best_alpha,
        "final_val_all_rmse": round(rmse(y_va_all, final_va_all), 3),
        "final_val_all_asym_mean": round(asymmetric_rul_score(y_va_all, final_va_all) / len(y_va_all), 3),
        "final_val_180cut_rmse": round(rmse(y_va_cut, final_va_cut), 3),
        "final_val_180cut_asym_sum": round(asymmetric_rul_score(y_va_cut, final_va_cut), 1),
        "range_breakdown": range_breakdown(y_va_all, final_va_all),
        "uncertainty_90pi": {
            "conformal_q90": round(q90, 3),
            "val_180cut_coverage": round(cov90_cut, 4),
            "val_180cut_avg_width_cycles": round(avg_width90, 2),
        },
        "alpha_tradeoff": alpha_table,
    }

    # Save test predictions (prediksi_rul.csv)
    sub = pd.DataFrame({"id_unit": te_units, "rul": np.round(final_te_pred, 3)})
    sub = sub.sort_values("id_unit").reset_index(drop=True)
    sub_path = os.path.join(PROJECT_DIR, "prediksi_rul.csv")
    sub.to_csv(sub_path, index=False)
    results["test_pred_summary"] = {
        "n_rows": int(len(sub)),
        "min": round(float(sub["rul"].min()), 2),
        "median": round(float(sub["rul"].median()), 2),
        "mean": round(float(sub["rul"].mean()), 2),
        "max": round(float(sub["rul"].max()), 2),
    }
    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_json(results, os.path.join(RESULTS_DIR, "task2_results.json"))

    plot_task2_results(arch_hists, y_va_cut, va_cut_mu, lo90_cut, hi90_cut, final_va_cut)
    print(f"[Task 2] Done in {results['elapsed_sec']}s! Final val_all RMSE={results['final_ensemble']['final_val_all_rmse']:.3f}, "
          f"180cut Asym={results['final_ensemble']['final_val_180cut_asym_sum']:.1f}")


def plot_task2_results(arch_hists, y_true, y_mu, lo90, hi90, y_final):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))
    # Panel A: Val RMSE curves across 3 DL architectures
    for arch, col, lbl in zip(["tcn", "bigru", "hybrid"], ["#ff7f0e", "#1f77b4", "#2ca02c"],
                              ["Dilated TCN", "BiGRU-Attention", "Hybrid CNN-BiGRU-Tr"]):
        h = arch_hists[arch]["val_rmse"]
        axes[0].plot(range(1, len(h) + 1), h, color=col, lw=1.5, label=f"{lbl} (min={min(h):.2f})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation RMSE (siklus)")
    axes[0].set_title("(a) Perbandingan Kurva Validasi 3 Arsitektur DL")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3)

    # Panel B: Predicted vs Actual RUL scatter on 180 val machines
    axes[1].scatter(y_true, y_final, alpha=0.6, s=16, color="#1f77b4", label="Prediksi Final")
    axes[1].plot([0, 150], [0, 150], "r--", lw=1.2, label="Ideal (y = x)")
    axes[1].set_xlabel("RUL Aktual (siklus)")
    axes[1].set_ylabel("RUL Prediksi (siklus)")
    axes[1].set_title("(b) Prediksi vs Aktual (180 Mesin Validasi)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel C: 90% Prediction Intervals sorted by true RUL (subset of 45 machines for clarity)
    order = np.argsort(y_true)[::4]
    x = np.arange(len(order))
    axes[2].fill_between(x, lo90[order], hi90[order], color="#aec7e8", alpha=0.65, label="90% PI (Conformal+MC)")
    axes[2].plot(x, y_mu[order], color="#1f77b4", lw=1.2, label="Mean Prediksi")
    axes[2].scatter(x, y_true[order], color="#d62728", s=14, zorder=4, label="RUL Aktual")
    axes[2].axhline(15, color="green", ls=":", lw=1.1, label="Ambang Servis (L90<=15)")
    axes[2].set_xlabel("Mesin Validasi (diurutkan berdasar RUL)")
    axes[2].set_ylabel("RUL (siklus)")
    axes[2].set_title("(c) Interval Prediksi 90% & Ambang Servis")
    axes[2].legend(fontsize=6.8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t2_rul_results.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    run_task2(args.epochs)
