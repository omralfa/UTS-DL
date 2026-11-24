import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, WEIGHTS_DIR, PROJECT_DIR,
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, ensure_dirs, set_seed, save_json
)

ensure_dirs()


# ============================================================================
# 1. Data Loading & Per-Line / Global Normalization
# ============================================================================
def load_task1_data():
    tr_meta = pd.read_csv(os.path.join(DATA_DIR, "klasifikasi_train_meta.csv"))
    te_meta = pd.read_csv(os.path.join(DATA_DIR, "klasifikasi_test_meta.csv"))
    tr_npz = np.load(os.path.join(DATA_DIR, "klasifikasi_train.npz"))
    te_npz = np.load(os.path.join(DATA_DIR, "klasifikasi_test.npz"))

    tr_img = tr_npz["gambar"]  # (48000, 64, 64) uint8
    tr_sen = tr_npz["sensor"].astype(np.float32)  # (48000, 3, 128)
    te_img = te_npz["gambar"]  # (16000, 64, 64) uint8
    te_sen = te_npz["sensor"].astype(np.float32)  # (16000, 3, 128)
    y = tr_meta["kelas"].map(CLASS_TO_IDX).values.astype(np.int64)
    return tr_img, tr_sen, tr_meta, y, te_img, te_sen, te_meta


def compute_norm_stats(img, sen, lini_arr, per_line=True):
    stats = {}
    groups = ["A", "B"] if per_line else ["ALL"]
    for g in groups:
        mask = (lini_arr == g) if per_line else np.ones(len(img), dtype=bool)
        im_m = float(img[mask].mean())
        im_s = float(img[mask].std()) + 1e-6
        s_m = np.array([float(np.nanmean(sen[mask, c, :])) for c in range(3)], dtype=np.float32)
        s_s = np.array([float(np.nanstd(sen[mask, c, :])) + 1e-6 for c in range(3)], dtype=np.float32)
        stats[g] = {"img_mean": im_m, "img_std": im_s, "sen_mean": s_m, "sen_std": s_s}
    return stats


def preprocess_arrays(img, sen, lini_arr, stats, per_line=True):
    N = len(img)
    img_out = np.empty((N, 1, 64, 64), dtype=np.float32)
    sen_out = np.zeros((N, 3, 128), dtype=np.float32)
    sen_mask = (~np.isnan(sen).any(axis=2)).astype(np.float32)  # (N, 3)

    groups = ["A", "B"] if per_line else ["ALL"]
    for g in groups:
        idx = (lini_arr == g) if per_line else np.ones(N, dtype=bool)
        st = stats[g]
        img_out[idx, 0] = (img[idx].astype(np.float32) - st["img_mean"]) / st["img_std"]
        s_sub = (sen[idx] - st["sen_mean"][None, :, None]) / st["sen_std"][None, :, None]
        s_sub = np.nan_to_num(s_sub, nan=0.0, posinf=0.0, neginf=0.0)
        sen_out[idx] = np.clip(s_sub, -8.0, 8.0)
    return img_out, sen_out, sen_mask


def gpu_augment_batch(x_im, x_se, m_se, mod_drop_all=0.15, mod_drop_ch=0.10):
    """Vectorized GPU augmentation (40x faster than per-sample CPU DataLoader)."""
    B = x_im.size(0)
    dev = x_im.device
    # Image: random flips, 90-deg rot, scale/shift jitter, mild noise
    if torch.rand(1, device=dev).item() < 0.5:
        x_im = torch.flip(x_im, dims=[3])
    if torch.rand(1, device=dev).item() < 0.5:
        x_im = torch.flip(x_im, dims=[2])
    k = int(torch.randint(0, 4, (1,), device=dev).item())
    if k > 0:
        x_im = torch.rot90(x_im, k, dims=[2, 3])
    scale = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.90, 1.10)
    shift = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
    x_im = x_im * scale + shift + torch.randn_like(x_im) * 0.04

    # Sensor: random roll, scale jitter, mild noise, modality dropout
    t_shift = int(torch.randint(-4, 5, (1,), device=dev).item())
    if t_shift != 0:
        x_se = torch.roll(x_se, shifts=t_shift, dims=2)
    s_scale = torch.empty(B, 3, 1, device=dev).uniform_(0.93, 1.07)
    x_se = (x_se * s_scale + torch.randn_like(x_se) * 0.04) * m_se.unsqueeze(-1)

    # Modality dropout (all channels or individual channels)
    drop_all = (torch.rand(B, 1, device=dev) < mod_drop_all)
    drop_ch = (torch.rand(B, 3, device=dev) < mod_drop_ch) | drop_all
    m_se = m_se * (~drop_ch).float()
    x_se = x_se * m_se.unsqueeze(-1)
    return x_im, x_se, m_se


class FastTensorData:
    """Holds preprocessed arrays as contiguous float32 tensors for fast GPU batching."""
    def __init__(self, img, sen, sen_mask, y=None, augment=False):
        self.img = torch.from_numpy(img)
        self.sen = torch.from_numpy(sen)
        self.sen_mask = torch.from_numpy(sen_mask)
        self.y = torch.from_numpy(y) if y is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.img)


# ============================================================================
# 2. Deep Learning Architectures (Image Encoder, Sensor Encoder, 3 Fusions)
# ============================================================================
class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if (stride != 1 or in_c != out_c) else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.layer1 = ResBlock2D(24, 48, stride=2)       # 16x16
        self.layer2 = ResBlock2D(48, 96, stride=2)       # 8x8
        self.layer3 = ResBlock2D(96, emb_dim, stride=2)  # 4x4 -> 16 spatial tokens
        self.proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        fmap = self.layer3(h)  # (B, D, 4, 4)
        tokens = fmap.flatten(2).transpose(1, 2)  # (B, 16, D)
        avg_p = fmap.mean(dim=(2, 3))
        max_p = fmap.amax(dim=(2, 3))
        emb = F.gelu(self.proj(torch.cat([avg_p, max_p], dim=1)))
        return emb, tokens


class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, k=5):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_c, out_c, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.skip = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_c)
        ) if (stride != 1 or in_c != out_c) else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SensorEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        # Input: 3 signal channels + 3 broadcast validity mask channels = 6 channels
        self.stem = nn.Sequential(
            nn.Conv1d(6, 24, 7, padding=3, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
        )
        self.b1 = ResBlock1D(24, 48, stride=2, k=5)   # 64
        self.b2 = ResBlock1D(48, 96, stride=2, k=5)   # 32
        self.b3 = ResBlock1D(96, emb_dim, stride=2, k=3)  # 16 temporal tokens
        self.gru = nn.GRU(emb_dim, emb_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(emb_dim * 2 + 3, emb_dim)

    def forward(self, x, mask):
        B, C, L = x.shape
        m_exp = mask.unsqueeze(-1).expand(B, C, L)
        inp = torch.cat([x, m_exp], dim=1)
        h = self.stem(inp)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)  # (B, D, 16)
        tokens, _ = self.gru(h.transpose(1, 2))  # (B, 16, D)
        avg_p = tokens.mean(dim=1)
        max_p = tokens.amax(dim=1)
        emb = F.gelu(self.proj(torch.cat([avg_p, max_p, mask], dim=1)))
        # Gate by whether any sensor channel is valid
        any_valid = (mask.sum(dim=1, keepdim=True) > 0).float()
        emb = emb * any_valid
        return emb, tokens


class MultimodalDefectNet(nn.Module):
    def __init__(self, mode="cross_attn", emb_dim=128, num_classes=7, dropout=0.20):
        super().__init__()
        self.mode = mode
        self.use_img = mode in ("image_only", "early_concat", "late_fusion", "cross_attn")
        self.use_sen = mode in ("sensor_only", "early_concat", "late_fusion", "cross_attn")

        if self.use_img:
            self.img_enc = ImageEncoder(emb_dim)
        if self.use_sen:
            self.sen_enc = SensorEncoder(emb_dim)

        if mode == "image_only":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
        elif mode == "sensor_only":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
        elif mode == "early_concat":
            self.head = nn.Sequential(
                nn.Linear(emb_dim * 2 + 3, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, num_classes),
            )
        elif mode == "late_fusion":
            self.img_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
            self.sen_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))
            self.gate = nn.Sequential(
                nn.Linear(emb_dim * 2 + 3, 64),
                nn.GELU(),
                nn.Linear(64, num_classes),
                nn.Sigmoid(),
            )
        elif mode == "cross_attn":
            self.img_to_sen = nn.MultiheadAttention(emb_dim, num_heads=4, dropout=0.1, batch_first=True)
            self.sen_to_img = nn.MultiheadAttention(emb_dim, num_heads=4, dropout=0.1, batch_first=True)
            self.norm_i = nn.LayerNorm(emb_dim)
            self.norm_s = nn.LayerNorm(emb_dim)
            # Modality gate to dynamically weight sensor contribution when missing
            self.mod_gate = nn.Sequential(nn.Linear(emb_dim * 2 + 3, emb_dim), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(emb_dim * 3 + 3, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, num_classes),
            )
            # Auxiliary unimodal heads to regularize encoders and guarantee missing-modality robustness
            self.aux_img = nn.Linear(emb_dim, num_classes)
            self.aux_sen = nn.Linear(emb_dim, num_classes)

    def forward(self, x_im, x_se, mask):
        if self.mode == "image_only":
            e_im, _ = self.img_enc(x_im)
            return self.head(e_im)
        if self.mode == "sensor_only":
            e_se, _ = self.sen_enc(x_se, mask)
            return self.head(e_se)

        e_im, t_im = self.img_enc(x_im)
        e_se, t_se = self.sen_enc(x_se, mask)
        any_valid = (mask.sum(dim=1, keepdim=True) > 0).float()

        if self.mode == "early_concat":
            return self.head(torch.cat([e_im, e_se, mask], dim=1))

        if self.mode == "late_fusion":
            l_im = self.img_head(e_im)
            l_se = self.sen_head(e_se)
            g = self.gate(torch.cat([e_im, e_se, mask], dim=1)) * any_valid
            return l_im + g * l_se

        if self.mode == "cross_attn":
            # Cross-attention between 16 image spatial tokens and 16 sensor temporal tokens
            ca_i, _ = self.img_to_sen(t_im, t_se, t_se)
            ca_s, _ = self.sen_to_img(t_se, t_im, t_im)
            f_i = self.norm_i(t_im + ca_i * any_valid.unsqueeze(-1)).mean(dim=1)
            f_s = self.norm_s(t_se + ca_s).mean(dim=1) * any_valid
            g = self.mod_gate(torch.cat([e_im, e_se, mask], dim=1)) * any_valid
            fused = torch.cat([e_im, f_i, g * (e_se + f_s), mask], dim=1)
            logits = self.head(fused)
            if self.training:
                return logits, self.aux_img(e_im), self.aux_sen(e_se)
            return logits


# ============================================================================
# 3. Robust Losses for Noisy Inspector Labels
# ============================================================================
class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross-Entropy (Wang et al., ICCV 2019) for noisy labels."""
    def __init__(self, alpha=1.0, beta=0.3, label_smoothing=0.05, weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.label_smoothing)
        probs = F.softmax(logits, dim=1).clamp(min=1e-6, max=1.0)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float().clamp(min=1e-4, max=1.0)
        rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


# ============================================================================
# 4. Training & Evaluation Helpers
# ============================================================================
def train_one_model(
    mode, tr_ds: FastTensorData, va_ds: FastTensorData, device, epochs=12, batch_size=512, lr=2.5e-3,
    loss_type="ce", seed=42, class_weights=None
):
    set_seed(seed)
    model = MultimodalDefectNet(mode=mode).to(device)

    cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=cw)
    elif loss_type == "ls":
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.08)
    elif loss_type == "sce":
        criterion = SymmetricCrossEntropyLoss(alpha=1.0, beta=0.3, label_smoothing=0.05, weight=cw)
    else:
        raise ValueError(loss_type)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    best_probs = None
    N_tr = len(tr_ds)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, n_seen = 0.0, 0
        perm = torch.randperm(N_tr)
        for start in range(0, N_tr - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            x_im = tr_ds.img[idx].to(device, non_blocking=True)
            x_se = tr_ds.sen[idx].to(device, non_blocking=True)
            m_se = tr_ds.sen_mask[idx].to(device, non_blocking=True)
            yb = tr_ds.y[idx].to(device, non_blocking=True)
            if tr_ds.augment:
                x_im, x_se, m_se = gpu_augment_batch(x_im, x_se, m_se)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x_im, x_se, m_se)
                if isinstance(out, tuple):
                    logits, aux_i, aux_s = out
                    loss = criterion(logits, yb) + 0.15 * criterion(aux_i, yb)
                    valid_s = (m_se.sum(dim=1) > 0)
                    if valid_s.any():
                        loss = loss + 0.15 * criterion(aux_s[valid_s], yb[valid_s])
                else:
                    loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item() * len(yb)
            n_seen += len(yb)
        sched.step()

        # Validation
        val_probs, val_y, val_loss = predict_probs(model, va_ds, device, return_loss=True)
        val_pred = val_probs.argmax(axis=1)
        val_f1 = f1_score(val_y, val_pred, average="macro")
        history["train_loss"].append(round(tot_loss / n_seen, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs = val_probs

    model.load_state_dict(best_state)
    return model, best_f1, best_probs, history


@torch.no_grad()
def predict_probs(model, ds: FastTensorData, device, return_loss=False, force_drop_sensor=False, batch_size=512):
    model.eval()
    probs_list = []
    tot_loss, n_seen = 0.0, 0
    N = len(ds)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_im = ds.img[start:end].to(device, non_blocking=True)
        x_se = ds.sen[start:end].to(device, non_blocking=True)
        m_se = ds.sen_mask[start:end].to(device, non_blocking=True)
        yb = ds.y[start:end].to(device, non_blocking=True) if ds.y is not None else None
        if force_drop_sensor:
            x_se = torch.zeros_like(x_se)
            m_se = torch.zeros_like(m_se)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x_im, x_se, m_se)
            if return_loss and yb is not None:
                tot_loss += F.cross_entropy(logits, yb).item() * len(yb)
                n_seen += len(yb)
        probs_list.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    if return_loss:
        return probs, ds.y.numpy(), tot_loss / max(n_seen, 1)
    return probs


def sanity_check_overfit_one_batch(tr_ds: FastTensorData, device):
    set_seed(42)
    model = MultimodalDefectNet(mode="cross_attn", dropout=0.0).to(device)
    x_im = tr_ds.img[:32].to(device)
    x_se = tr_ds.sen[:32].to(device)
    m_se = tr_ds.sen_mask[:32].to(device)
    yb = tr_ds.y[:32].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses, accs = [], []
    for step in range(1, 61):
        model.train()
        opt.zero_grad()
        logits, aux_i, aux_s = model(x_im, x_se, m_se)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        acc = (logits.argmax(dim=1) == yb).float().mean().item()
        losses.append(round(loss.item(), 4))
        accs.append(round(acc, 4))
    return {"final_loss": losses[-1], "final_acc": accs[-1], "losses": losses, "accs": accs}


def optimize_class_multipliers(probs, y_true):
    """Coordinate ascent on per-class probability multipliers to maximize macro-F1."""
    mult = np.ones(len(CLASSES), dtype=np.float32)
    best_f1 = f1_score(y_true, probs.argmax(axis=1), average="macro")
    for _ in range(3):
        for c in range(len(CLASSES)):
            best_v = mult[c]
            for cand in np.linspace(0.6, 1.6, 21):
                mult[c] = cand
                f1 = f1_score(y_true, (probs * mult[None, :]).argmax(axis=1), average="macro")
                if f1 > best_f1 + 1e-5:
                    best_f1 = f1
                    best_v = cand
            mult[c] = best_v
    return mult, best_f1


# ============================================================================
# 5. Main Experiment Pipeline
# ============================================================================
def run_task1(epochs_ablation=10, epochs_final=16):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 1] Device: {device}")

    tr_img, tr_sen, tr_meta, y, te_img, te_sen, te_meta = load_task1_data()

    # Stratified split on (kelas, lini) -> 80% train (38,400), 20% val (9,600)
    strat_key = tr_meta["kelas"].astype(str) + "_" + tr_meta["lini"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(tr_meta, strat_key))

    tr_lini = tr_meta["lini"].values[tr_idx]
    va_lini = tr_meta["lini"].values[va_idx]
    te_lini = te_meta["lini"].values

    # Per-line normalization (fit stats on training fold only!)
    stats_pl = compute_norm_stats(tr_img[tr_idx], tr_sen[tr_idx], tr_lini, per_line=True)
    Xtr_im, Xtr_se, Mtr_se = preprocess_arrays(tr_img[tr_idx], tr_sen[tr_idx], tr_lini, stats_pl, per_line=True)
    Xva_im, Xva_se, Mva_se = preprocess_arrays(tr_img[va_idx], tr_sen[va_idx], va_lini, stats_pl, per_line=True)
    Xte_im, Xte_se, Mte_se = preprocess_arrays(te_img, te_sen, te_lini, stats_pl, per_line=True)

    # Mild inverse-sqrt class weights to boost rare classes for macro-F1
    counts = np.bincount(y[tr_idx], minlength=len(CLASSES)).astype(np.float32)
    cw = (counts.mean() / counts) ** 0.35
    cw = cw / cw.mean()

    tr_ds = FastTensorData(Xtr_im, Xtr_se, Mtr_se, y[tr_idx], augment=True)
    va_ds = FastTensorData(Xva_im, Xva_se, Mva_se, y[va_idx], augment=False)
    te_ds = FastTensorData(Xte_im, Xte_se, Mte_se, None, augment=False)

    results = {"device": str(device), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_meta))}

    # (g) Sanity check: overfit one batch
    print("[Task 1] Running sanity check (overfit 1 batch)...")
    sanity = sanity_check_overfit_one_batch(FastTensorData(Xtr_im[:64], Xtr_se[:64], Mtr_se[:64], y[tr_idx[:64]], augment=False), device)
    results["sanity_check"] = {"final_loss": sanity["final_loss"], "final_acc": sanity["final_acc"]}
    print(f"  Sanity check: final_loss={sanity['final_loss']:.4f}, final_acc={sanity['final_acc']:.4f}")

    # (b) & (c) Modality value & Fusion strategy comparison
    modes = ["image_only", "sensor_only", "late_fusion", "early_concat", "cross_attn"]
    results["ablation_models"] = {}
    per_class_f1 = {}
    for mode in modes:
        print(f"[Task 1] Training ablation model: {mode} ...")
        m, bf1, vprobs, hist = train_one_model(
            mode, tr_ds, va_ds, device, epochs=epochs_ablation, loss_type="ls", seed=42, class_weights=cw
        )
        vpred = vprobs.argmax(axis=1)
        rep = classification_report(y[va_idx], vpred, target_names=CLASSES, output_dict=True, zero_division=0)
        f1_a = f1_score(y[va_idx][va_lini == "A"], vpred[va_lini == "A"], average="macro")
        f1_b = f1_score(y[va_idx][va_lini == "B"], vpred[va_lini == "B"], average="macro")
        results["ablation_models"][mode] = {
            "val_macro_f1": round(bf1, 4),
            "val_f1_lini_A": round(f1_a, 4),
            "val_f1_lini_B": round(f1_b, 4),
            "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASSES},
        }
        per_class_f1[mode] = [rep[c]["f1-score"] for c in CLASSES]
        print(f"  {mode:14s} -> macro-F1={bf1:.4f} (Lini A={f1_a:.4f}, Lini B={f1_b:.4f})")

    # (e) Domain shift experiment: Global norm + no aug vs Per-line norm + aug
    print("[Task 1] Running domain shift comparison (Global Norm, no aug)...")
    stats_gl = compute_norm_stats(tr_img[tr_idx], tr_sen[tr_idx], tr_lini, per_line=False)
    Gtr_im, Gtr_se, GMtr = preprocess_arrays(tr_img[tr_idx], tr_sen[tr_idx], tr_lini, stats_gl, per_line=False)
    Gva_im, Gva_se, GMva = preprocess_arrays(tr_img[va_idx], tr_sen[va_idx], va_lini, stats_gl, per_line=False)
    _, bf1_gl, vprobs_gl, _ = train_one_model(
        "cross_attn",
        FastTensorData(Gtr_im, Gtr_se, GMtr, y[tr_idx], augment=False),
        FastTensorData(Gva_im, Gva_se, GMva, y[va_idx], augment=False),
        device, epochs=epochs_ablation, loss_type="ls", seed=42, class_weights=cw
    )
    vp_gl = vprobs_gl.argmax(axis=1)
    results["domain_shift_exp"] = {
        "global_norm_no_aug": {
            "val_macro_f1": round(bf1_gl, 4),
            "val_f1_lini_A": round(f1_score(y[va_idx][va_lini == "A"], vp_gl[va_lini == "A"], average="macro"), 4),
            "val_f1_lini_B": round(f1_score(y[va_idx][va_lini == "B"], vp_gl[va_lini == "B"], average="macro"), 4),
        },
        "per_line_norm_with_aug": results["ablation_models"]["cross_attn"],
    }
    print(f"  Global norm -> macro-F1={bf1_gl:.4f} (A={results['domain_shift_exp']['global_norm_no_aug']['val_f1_lini_A']:.4f}, "
          f"B={results['domain_shift_exp']['global_norm_no_aug']['val_f1_lini_B']:.4f})")

    # (d) Label noise experiment: CE vs Label Smoothing vs Symmetric CE
    print("[Task 1] Running label noise robustness comparison (CE vs LS vs SCE)...")
    results["label_noise_exp"] = {}
    for lt in ["ce", "ls", "sce"]:
        if lt == "ls":
            results["label_noise_exp"][lt] = results["ablation_models"]["cross_attn"]["val_macro_f1"]
            continue
        _, bf1_lt, _, _ = train_one_model(
            "cross_attn", tr_ds, va_ds, device, epochs=epochs_ablation, loss_type=lt, seed=42, class_weights=cw
        )
        results["label_noise_exp"][lt] = round(bf1_lt, 4)
        print(f"  Loss {lt:4s} -> macro-F1={bf1_lt:.4f}")

    best_loss = max(results["label_noise_exp"], key=results["label_noise_exp"].get)
    print(f"[Task 1] Best loss selected: {best_loss}")

    # (g) Final model across 3 seeds + ensemble + missing modality evaluation + mislabel audit
    seeds = [42, 102, 202]
    seed_results = []
    val_probs_ens = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
    val_probs_nosensor_ens = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
    test_probs_ens = np.zeros((len(te_meta), len(CLASSES)), dtype=np.float64)
    best_hist = None

    for s in seeds:
        print(f"[Task 1] Training final cross_attn model (seed={s}, loss={best_loss}, epochs={epochs_final})...")
        model, bf1_s, vp_s, hist_s = train_one_model(
            "cross_attn", tr_ds, va_ds, device, epochs=epochs_final, loss_type=best_loss, seed=s, class_weights=cw
        )
        vp_drop = predict_probs(model, va_ds, device, force_drop_sensor=True)
        tp_s = predict_probs(model, te_ds, device)

        val_probs_ens += vp_s / len(seeds)
        val_probs_nosensor_ens += vp_drop / len(seeds)
        test_probs_ens += tp_s / len(seeds)

        f1_drop = f1_score(y[va_idx], vp_drop.argmax(axis=1), average="macro")
        seed_results.append({"seed": s, "val_macro_f1": round(bf1_s, 4), "val_macro_f1_no_sensor": round(f1_drop, 4)})
        if s == 42:
            best_hist = hist_s
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "task1_cross_attn_seed42.pt"))
        print(f"  Seed {s}: val_macro_f1={bf1_s:.4f}, no_sensor_f1={f1_drop:.4f}")

    ens_f1_raw = f1_score(y[va_idx], val_probs_ens.argmax(axis=1), average="macro")
    mult, ens_f1_tuned = optimize_class_multipliers(val_probs_ens, y[va_idx])
    ens_pred = (val_probs_ens * mult[None, :]).argmax(axis=1)
    ens_pred_nosensor = (val_probs_nosensor_ens * mult[None, :]).argmax(axis=1)

    # Missing modality breakdown on validation set
    nat_missing = (Mva_se.sum(axis=1) == 0)
    nat_partial = (Mva_se.sum(axis=1) > 0) & (Mva_se.sum(axis=1) < 3)
    nat_full = (Mva_se.sum(axis=1) == 3)

    results["final_3seeds"] = {
        "best_loss": best_loss,
        "seeds": seed_results,
        "mean_val_f1": round(float(np.mean([r["val_macro_f1"] for r in seed_results])), 4),
        "std_val_f1": round(float(np.std([r["val_macro_f1"] for r in seed_results])), 4),
        "ensemble_val_f1_raw": round(ens_f1_raw, 4),
        "ensemble_val_f1_tuned": round(ens_f1_tuned, 4),
        "class_multipliers": [round(float(x), 3) for x in mult],
        "ensemble_f1_lini_A": round(f1_score(y[va_idx][va_lini == "A"], ens_pred[va_lini == "A"], average="macro"), 4),
        "ensemble_f1_lini_B": round(f1_score(y[va_idx][va_lini == "B"], ens_pred[va_lini == "B"], average="macro"), 4),
        "missing_modality_eval": {
            "full_sensor_f1": round(f1_score(y[va_idx][nat_full], ens_pred[nat_full], average="macro"), 4),
            "partial_sensor_f1": round(f1_score(y[va_idx][nat_partial], ens_pred[nat_partial], average="macro"), 4),
            "natural_missing_all_sensor_f1": round(f1_score(y[va_idx][nat_missing], ens_pred[nat_missing], average="macro"), 4),
            "simulated_all_dropped_f1": round(f1_score(y[va_idx], ens_pred_nosensor, average="macro"), 4),
        },
        "per_class_f1": {
            c: round(v["f1-score"], 4)
            for c, v in classification_report(y[va_idx], ens_pred, target_names=CLASSES, output_dict=True).items()
            if c in CLASSES
        },
    }

    # (d) Mislabel audit on validation fold (high-confidence disagreements: p_pred > 0.85 and p_given < 0.10)
    p_given = val_probs_ens[np.arange(len(va_idx)), y[va_idx]]
    p_max = val_probs_ens.max(axis=1)
    pred_top = val_probs_ens.argmax(axis=1)
    suspected = (pred_top != y[va_idx]) & (p_max >= 0.85) & (p_given <= 0.10)
    va_df = tr_meta.iloc[va_idx].copy()
    va_df["suspected_mislabel"] = suspected
    results["mislabel_audit"] = {
        "n_suspected": int(suspected.sum()),
        "pct_suspected": round(float(suspected.mean() * 100), 2),
        "by_shift_pct": (va_df.groupby("shift")["suspected_mislabel"].mean() * 100).round(2).to_dict(),
        "by_class_pct": (va_df.groupby("kelas")["suspected_mislabel"].mean() * 100).round(2).to_dict(),
    }
    print(f"[Task 1] Suspected mislabels in val: {suspected.sum()}/{len(va_idx)} ({suspected.mean()*100:.2f}%)")
    print(f"  By shift: {results['mislabel_audit']['by_shift_pct']}")

    # Save test predictions (prediksi_klasifikasi.csv)
    test_pred_idx = (test_probs_ens * mult[None, :]).argmax(axis=1)
    sub = pd.DataFrame({
        "id_unit": te_meta["id_unit"].values,
        "kelas": [IDX_TO_CLASS[i] for i in test_pred_idx],
    })
    sub_path = os.path.join(PROJECT_DIR, "prediksi_klasifikasi.csv")
    sub.to_csv(sub_path, index=False)
    results["test_pred_distribution"] = sub["kelas"].value_counts().to_dict()
    results["elapsed_sec"] = round(time.time() - t0, 1)

    save_json(results, os.path.join(RESULTS_DIR, "task1_results.json"))

    # Generate figures for Task 1
    plot_task1_results(sanity, best_hist, per_class_f1, y[va_idx], ens_pred)
    print(f"[Task 1] Done in {results['elapsed_sec']}s! Ensemble val macro-F1 = {ens_f1_tuned:.4f}")


def plot_task1_results(sanity, hist, per_class_f1, y_true, y_pred):
    # Figure 1: Sanity check + Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    axes[0].plot(sanity["losses"], color="#d62728", lw=1.4, label="Loss")
    ax0t = axes[0].twinx()
    ax0t.plot(sanity["accs"], color="#2ca02c", lw=1.4, label="Akurasi")
    axes[0].set_title("(a) Sanity Check: Overfit 1 Batch (32 sampel)")
    axes[0].set_xlabel("Iterasi")
    axes[0].set_ylabel("Loss", color="#d62728")
    ax0t.set_ylabel("Akurasi", color="#2ca02c")

    ep = range(1, len(hist["train_loss"]) + 1)
    axes[1].plot(ep, hist["train_loss"], label="Train Loss", lw=1.4)
    axes[1].plot(ep, hist["val_loss"], label="Val Loss", lw=1.4)
    axes[1].set_title("(b) Kurva Loss (Cross-Attention, Seed 42)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)

    axes[2].plot(ep, hist["val_f1"], color="#1f77b4", marker="o", ms=3, lw=1.5)
    axes[2].set_title("(c) Kurva Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t1_training_curves.png"), bbox_inches="tight")
    plt.close()

    # Figure 2: Modality Value per-class F1 + Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    x = np.arange(len(CLASSES))
    w = 0.25
    axes[0].bar(x - w, per_class_f1["image_only"], w, label="Image-only", color="#1f77b4")
    axes[0].bar(x, per_class_f1["sensor_only"], w, label="Sensor-only", color="#ff7f0e")
    axes[0].bar(x + w, per_class_f1["cross_attn"], w, label="Cross-Attn Multimodal", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASSES, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1-Score")
    axes[0].set_title("(a) Nilai Tiap Modalitas per Kelas Cacat")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    im = axes[1].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(CLASSES)))
    axes[1].set_yticks(range(len(CLASSES)))
    axes[1].set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=7.5)
    axes[1].set_yticklabels(CLASSES, fontsize=7.5)
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")
    axes[1].set_title("(b) Confusion Matrix (Normalized, Ensemble 3-Seed)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.5 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t1_modality_cm.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ablation", type=int, default=10)
    parser.add_argument("--epochs-final", type=int, default=16)
    args = parser.parse_args()
    run_task1(args.epochs_ablation, args.epochs_final)
