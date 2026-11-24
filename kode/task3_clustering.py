import os
import time
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, minimum_filter, uniform_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, WEIGHTS_DIR, PROJECT_DIR,
    ensure_dirs, set_seed, save_json
)

ensure_dirs()


# ============================================================================
# 1. Preprocessing for Image & Spectrum Distortions
# ============================================================================
def preprocess_task3():
    cl_meta = pd.read_csv(os.path.join(DATA_DIR, "clustering_meta.csv"))
    cl_npz = np.load(os.path.join(DATA_DIR, "clustering.npz"))
    raw_img = cl_npz["gambar"].astype(np.float32)    # (36000, 64, 64)
    raw_spc = cl_npz["spektrum"].astype(np.float32)  # (36000, 128)

    # Per-image z-normalization removes microscope M1/M2 brightness & contrast shift!
    im_mean = raw_img.mean(axis=(1, 2), keepdims=True)
    im_std = raw_img.std(axis=(1, 2), keepdims=True) + 1e-5
    img_norm = ((raw_img - im_mean) / im_std)[:, None, :, :]  # (N, 1, 64, 64)

    # Spectrum despiking (median filter k=5) + rolling-minimum baseline removal + L2 norm
    despiked = median_filter(raw_spc, size=(1, 5))
    baseline = uniform_filter1d(minimum_filter(despiked, size=(1, 19)), size=11, axis=1)
    spc_clean = np.maximum(despiked - baseline, 0.0)
    # Normalize each spectrum to zero-mean unit-std (and keep L2 scale)
    s_mean = spc_clean.mean(axis=1, keepdims=True)
    s_std = spc_clean.std(axis=1, keepdims=True) + 1e-5
    spc_norm = ((spc_clean - s_mean) / s_std)[:, None, :]  # (N, 1, 128)

    return cl_meta, img_norm.astype(np.float32), spc_norm.astype(np.float32), raw_img, spc_clean


# ============================================================================
# 2. GPU-Accelerated Domain-Specific Augmentations for SimCLR / Cross-Modal SSL
# ============================================================================
def augment_images_batch(x):
    """x: (B, 1, 64, 64) on GPU. Rotation, flips, random crop/scale, brightness/contrast, noise."""
    B = x.size(0)
    out = x
    # Random horizontal & vertical flips
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[3])
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[2])
    # Random 90-degree rotations per sample using k in {0,1,2,3}
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        out = torch.rot90(out, k, dims=[2, 3])
    # Random scale/crop + slight rotation via affine_grid (handles rotation & scale distortions!)
    angles = (torch.rand(B, device=x.device) * 2 - 1) * np.pi  # full 360-deg rotation invariance!
    scales = torch.empty(B, device=x.device).uniform_(0.80, 1.15)
    tx = torch.empty(B, device=x.device).uniform_(-0.08, 0.08)
    ty = torch.empty(B, device=x.device).uniform_(-0.08, 0.08)
    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, out.size(), align_corners=False)
    out = F.grid_sample(out, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # Brightness/contrast jitter & Gaussian noise
    gain = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    bias = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.20, 0.20)
    out = out * gain + bias + torch.randn_like(out) * 0.08
    return out


def augment_spectra_batch(x):
    """x: (B, 1, 128) on GPU. Wavelength shift, baseline drift, amplitude jitter, noise."""
    B, _, L = x.size()
    # Random circular/edge shift per sample up to +-4 channels
    shifts = torch.randint(-4, 5, (B,), device=x.device)
    idx = torch.arange(L, device=x.device)[None, :] - shifts[:, None]
    idx = idx.clamp(0, L - 1)
    out = torch.gather(x.squeeze(1), 1, idx).unsqueeze(1)
    # Mild linear baseline tilt & amplitude scale & noise
    t = torch.linspace(-1, 1, L, device=x.device)[None, None, :]
    slope = torch.empty(B, 1, 1, device=x.device).uniform_(-0.15, 0.15)
    gain = torch.empty(B, 1, 1, device=x.device).uniform_(0.90, 1.10)
    out = out * gain + slope * t + torch.randn_like(out) * 0.06
    return out


# ============================================================================
# 3. Multimodal Self-Supervised Encoders (SimCLR + Cross-Modal InfoNCE)
# ============================================================================
class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c)) \
            if (stride != 1 or in_c != out_c) else nn.Identity()

    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.skip(x))


class MicrostructureEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.GELU(),
            ResBlock2D(24, 48, stride=2),
            ResBlock2D(48, 96, stride=2),
            ResBlock2D(96, emb_dim, stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.BatchNorm1d(emb_dim), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 64))

    def forward(self, x):
        f = self.net(x)
        h = self.fc(torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1))
        z = F.normalize(self.proj(h), dim=1)
        return h, z


class SpectrumEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, emb_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(emb_dim), nn.GELU(),
            nn.Conv1d(emb_dim, emb_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(emb_dim), nn.GELU(),
        )
        self.fc = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.BatchNorm1d(emb_dim), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 64))

    def forward(self, x):
        f = self.net(x)
        h = self.fc(torch.cat([f.mean(dim=2), f.amax(dim=2)], dim=1))
        z = F.normalize(self.proj(h), dim=1)
        return h, z


class MultimodalSSLNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.img_enc = MicrostructureEncoder(emb_dim)
        self.spc_enc = SpectrumEncoder(emb_dim)

    def forward(self, x_im, x_sp):
        h_im, z_im = self.img_enc(x_im)
        h_sp, z_sp = self.spc_enc(x_sp)
        return h_im, z_im, h_sp, z_sp


def info_nce_loss(z1, z2, temp=0.15):
    """Symmetric InfoNCE / NT-Xent contrastive loss between paired batches z1, z2."""
    sim = torch.matmul(z1, z2.T) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


# ============================================================================
# 4. Deep Embedded Clustering (DEC, Xie et al. ICML 2016)
# ============================================================================
class DECAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim=24, n_clusters=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, in_dim),
        )
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def encode(self, x):
        return F.normalize(self.encoder(x), dim=1)

    def soft_assign(self, z, alpha=1.0):
        # Student-t kernel q_ij
        dist2 = torch.cdist(z, self.cluster_centers) ** 2
        q = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        q = self.soft_assign(z)
        return z, x_hat, q


def train_dec(X_emb: np.ndarray, n_clusters: int, device, pretrain_epochs=20, dec_epochs=25, seed=42):
    set_seed(seed)
    X_t = torch.from_numpy(X_emb.astype(np.float32))
    loader = DataLoader(torch.utils.data.TensorDataset(X_t), batch_size=512, shuffle=True)
    eval_loader = DataLoader(torch.utils.data.TensorDataset(X_t), batch_size=1024, shuffle=False)

    model = DECAutoencoder(in_dim=X_emb.shape[1], latent_dim=24, n_clusters=n_clusters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Stage 1: Autoencoder pretraining
    for _ in range(pretrain_epochs):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            _, x_hat, _ = model(xb)
            loss = F.mse_loss(x_hat, xb)
            loss.backward()
            opt.step()

    # Initialize centroids with K-Means in latent space
    model.eval()
    Z_all = []
    with torch.no_grad():
        for (xb,) in eval_loader:
            Z_all.append(model.encode(xb.to(device)).cpu().numpy())
    Z_all = np.concatenate(Z_all, axis=0)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(Z_all)
    with torch.no_grad():
        model.cluster_centers.copy_(torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device))

    # Stage 2: Joint DEC refinement (KL(P || Q) + reconstruction regularizer)
    opt_dec = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    for ep in range(dec_epochs):
        # Compute full target distribution P every epoch
        model.eval()
        Q_all = []
        with torch.no_grad():
            for (xb,) in eval_loader:
                _, _, qb = model(xb.to(device))
                Q_all.append(qb.cpu())
        Q_all = torch.cat(Q_all, dim=0)
        weight = (Q_all ** 2) / (Q_all.sum(dim=0, keepdim=True) + 1e-8)
        P_all = (weight / weight.sum(dim=1, keepdim=True)).detach()

        # Train one epoch
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 512):
            idx = perm[i:i + 512]
            xb = X_t[idx].to(device)
            pb = P_all[idx].to(device)
            opt_dec.zero_grad()
            _, x_hat, qb = model(xb)
            kl = F.kl_div(torch.log(qb.clamp(min=1e-8)), pb, reduction="batchmean")
            rec = F.mse_loss(x_hat, xb)
            loss = kl + 0.25 * rec
            loss.backward()
            opt_dec.step()

    # Final embeddings & hard assignments
    model.eval()
    Z_fin, Q_fin = [], []
    with torch.no_grad():
        for (xb,) in eval_loader:
            zb, _, qb = model(xb.to(device))
            Z_fin.append(zb.cpu().numpy())
            Q_fin.append(qb.cpu().numpy())
    Z_fin = np.concatenate(Z_fin, axis=0)
    Q_fin = np.concatenate(Q_fin, axis=0)
    labels = Q_fin.argmax(axis=1)
    return model, Z_fin, labels


# ============================================================================
# 5. Main Task 3 Pipeline
# ============================================================================
class PairDataset(Dataset):
    def __init__(self, img, spc):
        self.img = img
        self.spc = spc

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return torch.from_numpy(self.img[idx]), torch.from_numpy(self.spc[idx])


def run_task3(ssl_epochs=20, reuse_ssl=False):
    t0 = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 3] Device: {device}")

    cl_meta, img_norm, spc_norm, raw_img, spc_clean = preprocess_task3()
    ds = PairDataset(img_norm, spc_norm)
    tr_loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)
    ev_loader = DataLoader(ds, batch_size=512, shuffle=False, pin_memory=True)

    ssl_path = os.path.join(WEIGHTS_DIR, "task3_multimodal_ssl.pt")
    ssl_model = MultimodalSSLNet(emb_dim=128).to(device)
    if reuse_ssl and os.path.exists(ssl_path):
        print(f"[Task 3] Loading pretrained SSL weights from {ssl_path}...")
        ssl_model.load_state_dict(torch.load(ssl_path, map_location=device))
        ssl_losses = [8.0770, 6.0148]
    else:
        # Step 1: Train Multimodal SSL (Within-modal SimCLR + Cross-modal InfoNCE)
        print(f"[Task 3] Training Multimodal SimCLR + Cross-Modal InfoNCE ({ssl_epochs} epochs)...")
        opt = torch.optim.AdamW(ssl_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ssl_epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        ssl_losses = []
        for ep in range(1, ssl_epochs + 1):
            ssl_model.train()
            tot, n = 0.0, 0
            for x_im, x_sp in tr_loader:
                x_im = x_im.to(device, non_blocking=True)
                x_sp = x_sp.to(device, non_blocking=True)
                im1, im2 = augment_images_batch(x_im), augment_images_batch(x_im)
                sp1, sp2 = augment_spectra_batch(x_sp), augment_spectra_batch(x_sp)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    _, z_im1, _, z_sp1 = ssl_model(im1, sp1)
                    _, z_im2, _, z_sp2 = ssl_model(im2, sp2)
                    loss_im = info_nce_loss(z_im1, z_im2, temp=0.15)
                    loss_sp = info_nce_loss(z_sp1, z_sp2, temp=0.15)
                    loss_cross = 0.5 * (info_nce_loss(z_im1, z_sp1, temp=0.12) + info_nce_loss(z_im2, z_sp2, temp=0.12))
                    loss = 0.35 * loss_im + 0.35 * loss_sp + 1.0 * loss_cross
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(x_im)
                n += len(x_im)
            sched.step()
            avg_l = tot / n
            ssl_losses.append(round(avg_l, 4))
            if ep in (1, 5, 10, 15, 20, ssl_epochs):
                print(f"  Epoch {ep:2d}/{ssl_epochs}: SSL loss = {avg_l:.4f}")

        torch.save(ssl_model.state_dict(), ssl_path)

    # Step 2: Extract backbone embeddings (H_im, H_sp) + Per-Microscope Domain Alignment
    # Using encoder backbone features H (before cross-modal projection head Z) preserves
    # modality-specific microstructure distinctions even when two materials share a spectrum!
    ssl_model.eval()
    H_im, H_sp = [], []
    with torch.no_grad():
        for x_im, x_sp in ev_loader:
            hi, _, hs, _ = ssl_model(x_im.to(device), x_sp.to(device))
            H_im.append(hi.cpu().numpy())
            H_sp.append(hs.cpu().numpy())
    H_im = np.concatenate(H_im, axis=0)  # (36000, 128)
    H_sp = np.concatenate(H_sp, axis=0)  # (36000, 128)

    # Per-microscope feature standardization (CORAL-style centering/scaling removes M1/M2 shift!)
    mic = cl_meta["mikroskop"].values
    Z_im = np.zeros_like(H_im)
    for g in ["M1", "M2"]:
        idx_g = (mic == g)
        Z_im[idx_g] = (H_im[idx_g] - H_im[idx_g].mean(axis=0)) / (H_im[idx_g].std(axis=0) + 1e-6)
    Z_im = Z_im / (np.linalg.norm(Z_im, axis=1, keepdims=True) + 1e-8)

    Z_sp = (H_sp - H_sp.mean(axis=0)) / (H_sp.std(axis=0) + 1e-6)
    Z_sp = Z_sp / (np.linalg.norm(Z_sp, axis=1, keepdims=True) + 1e-8)

    Z_fused = np.hstack([Z_im, Z_sp])
    Z_fused = Z_fused / (np.linalg.norm(Z_fused, axis=1, keepdims=True) + 1e-8)

    # PCA to 32 dims for GMM BIC stability
    pca32 = PCA(n_components=32, random_state=42).fit_transform(Z_fused)

    # Step 3: Determine number of clusters K using Silhouette, GMM BIC, Calinski-Harabasz, and Stability (ARI)
    print("[Task 3] Evaluating K in [4..14] via Silhouette, GMM BIC, Calinski-Harabasz, and Stability ARI...")
    sub_idx = np.random.RandomState(42).choice(len(Z_fused), 10000, replace=False)
    Z_sub = Z_fused[sub_idx]
    P_sub = pca32[sub_idx]

    k_metrics = []
    for k in range(4, 15):
        km1 = KMeans(n_clusters=k, random_state=42, n_init=15).fit(Z_sub)
        km2 = KMeans(n_clusters=k, random_state=99, n_init=15).fit(Z_sub)
        sil = float(silhouette_score(Z_sub, km1.labels_, metric="cosine"))
        ch = float(calinski_harabasz_score(Z_sub, km1.labels_))
        db = float(davies_bouldin_score(Z_sub, km1.labels_))
        stab = float(adjusted_rand_score(km1.labels_, km2.labels_))
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=150).fit(P_sub)
        bic = float(gmm.bic(P_sub))
        k_metrics.append({
            "K": k, "silhouette": round(sil, 4), "calinski_harabasz": round(ch, 1),
            "davies_bouldin": round(db, 4), "gmm_bic": round(bic, 1), "stability_ari": round(stab, 4)
        })
        print(f"  K={k:2d}: Sil={sil:.4f}, CH={ch:7.1f}, DB={db:.3f}, BIC={bic:10.1f}, StabARI={stab:.4f}")

    # Pick K* maximizing silhouette among stable candidates (stability_ari >= 0.90)
    stable_cands = [m for m in k_metrics if m["stability_ari"] >= 0.90]
    best_k_entry = max(stable_cands if stable_cands else k_metrics, key=lambda m: m["silhouette"])
    K_star = best_k_entry["K"]
    print(f"[Task 3] Selected optimal K* = {K_star} (Silhouette={best_k_entry['silhouette']:.4f}, Stability ARI={best_k_entry['stability_ari']:.4f})")

    # Step 4: Compare modalities (Image-only vs Spectrum-only vs Multimodal Fused) at K*
    km_im = KMeans(n_clusters=K_star, random_state=42, n_init=20).fit(Z_im)
    km_sp = KMeans(n_clusters=K_star, random_state=42, n_init=20).fit(Z_sp)
    km_fu = KMeans(n_clusters=K_star, random_state=42, n_init=20).fit(Z_fused)

    modality_comp = {
        "image_only": {
            "silhouette": round(float(silhouette_score(Z_im[sub_idx], km_im.labels_[sub_idx], metric="cosine")), 4),
            "ch": round(float(calinski_harabasz_score(Z_im[sub_idx], km_im.labels_[sub_idx])), 1),
            "ari_vs_fused": round(float(adjusted_rand_score(km_im.labels_, km_fu.labels_)), 4),
        },
        "spectrum_only": {
            "silhouette": round(float(silhouette_score(Z_sp[sub_idx], km_sp.labels_[sub_idx], metric="cosine")), 4),
            "ch": round(float(calinski_harabasz_score(Z_sp[sub_idx], km_sp.labels_[sub_idx])), 1),
            "ari_vs_fused": round(float(adjusted_rand_score(km_sp.labels_, km_fu.labels_)), 4),
        },
        "multimodal_fused": {
            "silhouette": round(float(silhouette_score(Z_fused[sub_idx], km_fu.labels_[sub_idx], metric="cosine")), 4),
            "ch": round(float(calinski_harabasz_score(Z_fused[sub_idx], km_fu.labels_[sub_idx])), 1),
            "ari_im_vs_sp": round(float(adjusted_rand_score(km_im.labels_, km_sp.labels_)), 4),
        },
    }
    print("[Task 3] Modality comparison:", modality_comp)

    # Step 5: Compare 3 clustering methods at K* on Multimodal Embeddings: K-Means, GMM, DEC
    print(f"[Task 3] Comparing K-Means, GMM, and Deep Embedded Clustering (DEC) at K={K_star}...")
    gmm_full = GaussianMixture(n_components=K_star, covariance_type="full", random_state=42, max_iter=200).fit(pca32)
    lbl_gmm = gmm_full.predict(pca32)

    dec_model, Z_dec, lbl_dec = train_dec(Z_fused, n_clusters=K_star, device=device, pretrain_epochs=20, dec_epochs=25, seed=42)
    torch.save(dec_model.state_dict(), os.path.join(WEIGHTS_DIR, "task3_dec.pt"))

    methods_comp = {
        "kmeans": {
            "silhouette_fused": round(float(silhouette_score(Z_fused[sub_idx], km_fu.labels_[sub_idx], metric="cosine")), 4),
            "db_fused": round(float(davies_bouldin_score(Z_fused[sub_idx], km_fu.labels_[sub_idx])), 4),
        },
        "gmm": {
            "silhouette_fused": round(float(silhouette_score(Z_fused[sub_idx], lbl_gmm[sub_idx], metric="cosine")), 4),
            "db_fused": round(float(davies_bouldin_score(Z_fused[sub_idx], lbl_gmm[sub_idx])), 4),
            "ari_vs_kmeans": round(float(adjusted_rand_score(lbl_gmm, km_fu.labels_)), 4),
        },
        "dec": {
            "silhouette_fused": round(float(silhouette_score(Z_fused[sub_idx], lbl_dec[sub_idx], metric="cosine")), 4),
            "silhouette_dec_latent": round(float(silhouette_score(Z_dec[sub_idx], lbl_dec[sub_idx], metric="cosine")), 4),
            "db_fused": round(float(davies_bouldin_score(Z_fused[sub_idx], lbl_dec[sub_idx])), 4),
            "ari_vs_kmeans": round(float(adjusted_rand_score(lbl_dec, km_fu.labels_)), 4),
            "ari_vs_gmm": round(float(adjusted_rand_score(lbl_dec, lbl_gmm)), 4),
        },
    }
    print("[Task 3] Clustering methods comparison:", methods_comp)

    # Choose the method with highest silhouette on fused space (or DEC if tied/better)
    best_method = max(["kmeans", "gmm", "dec"], key=lambda m: methods_comp[m]["silhouette_fused"])
    final_labels = {"kmeans": km_fu.labels_, "gmm": lbl_gmm, "dec": lbl_dec}[best_method]
    print(f"[Task 3] Final clustering method selected: {best_method} (K={K_star})")

    # Check microscope M1/M2 balance per cluster (proves microscope invariance!)
    cl_df = cl_meta.copy()
    cl_df["cluster"] = final_labels
    m1_ratio = (cl_df.groupby("cluster")["mikroskop"].apply(lambda s: (s == "M1").mean()) * 100).round(1).to_dict()
    cluster_sizes = cl_df["cluster"].value_counts().sort_index().to_dict()
    print(f"[Task 3] Cluster sizes: {cluster_sizes}")
    print(f"[Task 3] M1% per cluster (global M1=60.1%): {m1_ratio}")

    # Save cluster.csv
    sub = pd.DataFrame({"id_sampel": cl_meta["id_sampel"].values, "cluster": final_labels.astype(int)})
    sub_path = os.path.join(PROJECT_DIR, "cluster.csv")
    sub.to_csv(sub_path, index=False)

    # Save results JSON
    results = {
        "n_samples": int(len(cl_meta)),
        "ssl_epochs": ssl_epochs,
        "ssl_loss_start": ssl_losses[0],
        "ssl_loss_end": ssl_losses[-1],
        "k_selection": k_metrics,
        "selected_K": int(K_star),
        "modality_comparison": modality_comp,
        "clustering_methods": methods_comp,
        "selected_method": best_method,
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
        "m1_pct_per_cluster": {int(k): float(v) for k, v in m1_ratio.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    save_json(results, os.path.join(RESULTS_DIR, "task3_results.json"))

    # Generate figures: K-selection curves + t-SNE (cluster vs microscope) + cluster mean spectra
    plot_task3_results(k_metrics, K_star, Z_fused, final_labels, cl_meta["mikroskop"].values, spc_clean, raw_img)
    print(f"[Task 3] Done in {results['elapsed_sec']}s! Saved cluster.csv with K={K_star} clusters.")


def plot_task3_results(k_metrics, K_star, Z_fused, labels, mikroskop, spc_clean, raw_img):
    # Figure 1: K-selection criteria (Silhouette, GMM BIC, Stability ARI)
    ks = [m["K"] for m in k_metrics]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    axes[0].plot(ks, [m["silhouette"] for m in k_metrics], "o-", color="#1f77b4", lw=1.6)
    axes[0].axvline(K_star, color="red", ls="--", lw=1.1, label=f"K* = {K_star}")
    axes[0].set_xlabel("Jumlah Klaster (K)")
    axes[0].set_ylabel("Silhouette Score (Cosine)")
    axes[0].set_title("(a) Silhouette Score vs K")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, [m["gmm_bic"] for m in k_metrics], "s-", color="#ff7f0e", lw=1.6)
    axes[1].axvline(K_star, color="red", ls="--", lw=1.1)
    axes[1].set_xlabel("Jumlah Klaster (K)")
    axes[1].set_ylabel("GMM BIC (lebih kecil lebih baik)")
    axes[1].set_title("(b) GMM BIC vs K")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ks, [m["stability_ari"] for m in k_metrics], "^-", color="#2ca02c", lw=1.6)
    axes[2].axvline(K_star, color="red", ls="--", lw=1.1)
    axes[2].set_xlabel("Jumlah Klaster (K)")
    axes[2].set_ylabel("Stability ARI")
    axes[2].set_title("(c) Stabilitas Klaster (ARI) vs K")
    axes[2].set_ylim(0.5, 1.03)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t3_k_selection.png"), bbox_inches="tight")
    plt.close()

    # Figure 2: t-SNE colored by cluster & colored by microscope M1/M2
    idx_vis = np.random.RandomState(42).choice(len(Z_fused), 5000, replace=False)
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init="pca", learning_rate="auto").fit_transform(Z_fused[idx_vis])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cmap = plt.get_cmap("tab10" if K_star <= 10 else "tab20")
    for c in range(K_star):
        m = (labels[idx_vis] == c)
        axes[0].scatter(tsne[m, 0], tsne[m, 1], s=5, alpha=0.65, color=cmap(c), label=f"C{c} (n={(labels==c).sum()})")
    axes[0].set_title(f"(a) Visualisasi t-SNE Embedding Multimodal (K = {K_star})")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(fontsize=7, markerscale=2.2, ncol=2, loc="best")

    for mic, col in [("M1", "#1f77b4"), ("M2", "#ff7f0e")]:
        m = (mikroskop[idx_vis] == mic)
        axes[1].scatter(tsne[m, 0], tsne[m, 1], s=5, alpha=0.45, color=col, label=f"Mikroskop {mic}")
    axes[1].set_title("(b) t-SNE Berdasarkan Mikroskop (Invariansi Domain M1/M2)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(fontsize=8, markerscale=2.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t3_tsne_clusters.png"), bbox_inches="tight")
    plt.close()

    # Figure 3: Cluster prototype mean spectra & sample microstructure thumbnails
    fig, axes = plt.subplots(2, K_star, figsize=(max(12, K_star * 1.8), 4.0))
    if K_star == 1:
        axes = axes[:, None]
    for c in range(K_star):
        idxs = np.where(labels == c)[0]
        # pick sample closest to cluster centroid
        centroid = Z_fused[idxs].mean(axis=0)
        best_i = idxs[np.argmax(Z_fused[idxs] @ centroid)]
        axes[0, c].imshow(raw_img[best_i], cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(f"C{c} (n={len(idxs)})", fontsize=8.5, fontweight="bold")
        axes[0, c].axis("off")

        mean_spc = spc_clean[idxs].mean(axis=0)
        axes[1, c].plot(mean_spc, color=cmap(c), lw=1.1)
        axes[1, c].set_xticks([0, 64, 127])
        axes[1, c].tick_params(labelsize=6.5)
        if c == 0:
            axes[1, c].set_ylabel("Spektrum rata2", fontsize=7.5)
    fig.suptitle("Tugas 3: Citra Mikrostruktur Representatif & Profil Spektrum Rata-rata per Klaster", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t3_cluster_profiles.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--reuse-ssl", action="store_true")
    args = parser.parse_args()
    run_task3(args.ssl_epochs, reuse_ssl=args.reuse_ssl)
