import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import DATA_DIR, FIGURES_DIR, CLASSES, ensure_dirs

ensure_dirs()
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 180,
})


def plot_task1_eda():
    tr_meta = pd.read_csv(os.path.join(DATA_DIR, "klasifikasi_train_meta.csv"))
    te_meta = pd.read_csv(os.path.join(DATA_DIR, "klasifikasi_test_meta.csv"))
    tr_npz = np.load(os.path.join(DATA_DIR, "klasifikasi_train.npz"))
    imgs = tr_npz["gambar"]
    sens = tr_npz["sensor"].astype(np.float32)

    # 1. Sample image + sensor per class
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.2))
    for i, cls in enumerate(CLASSES):
        sub = tr_meta[(tr_meta["kelas"] == cls) & (tr_meta["lini"] == "A")].index.values
        # pick one without NaN sensor
        idx = sub[0]
        for cand in sub[:50]:
            if not np.isnan(sens[cand]).any():
                idx = cand
                break
        axes[0, i].imshow(imgs[idx], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(cls, fontweight="bold")
        axes[0, i].axis("off")

        # plot normalized 3 sensor channels
        s = sens[idx]
        t = np.arange(128)
        for ch, col, lbl in zip(range(3), ["#d62728", "#1f77b4", "#2ca02c"], ["Suhu", "Getaran", "Arus"]):
            s_norm = (s[ch] - np.mean(s[ch])) / (np.std(s[ch]) + 1e-6)
            axes[1, i].plot(t, s_norm + ch * 2.5, color=col, lw=0.8, label=lbl if i == 0 else None)
        axes[1, i].set_xticks([0, 64, 127])
        axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Sinyal (norm)")
            axes[1, i].legend(loc="upper right", fontsize=6, frameon=False)
    fig.suptitle("Tugas 1: Contoh Citra Permukaan (Atas) dan Sinyal Sensor 3-Kanal (Bawah) per Kelas", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t1_eda_samples.png"), bbox_inches="tight")
    plt.close()

    # 2. Class distribution & Domain shift (Lini A vs B)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    # Panel A: Class counts
    counts = tr_meta["kelas"].value_counts().reindex(CLASSES)
    bars = axes[0].bar(range(len(CLASSES)), counts.values, color="#2b5c8f")
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=30, ha="right")
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].set_title("(a) Distribusi Kelas (Train)")
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width()/2, v + 350, f"{v/len(tr_meta)*100:.1f}%", ha="center", fontsize=7)

    # Panel B: Lini A vs B shift train vs test
    tr_l = tr_meta["lini"].value_counts(normalize=True).reindex(["A", "B"]) * 100
    te_l = te_meta["lini"].value_counts(normalize=True).reindex(["A", "B"]) * 100
    x = np.arange(2)
    w = 0.35
    axes[1].bar(x - w/2, tr_l.values, w, label="Train (N=48.000)", color="#1f77b4")
    axes[1].bar(x + w/2, te_l.values, w, label="Test (N=16.000)", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Lini A (M01-M06)", "Lini B (M07-M12)"])
    axes[1].set_ylabel("Proporsi (%)")
    axes[1].set_title("(b) Domain Shift Proporsi Lini")
    axes[1].legend(fontsize=8)
    for xi, v1, v2 in zip(x, tr_l.values, te_l.values):
        axes[1].text(xi - w/2, v1 + 1.5, f"{v1:.1f}%", ha="center", fontsize=7.5)
        axes[1].text(xi + w/2, v2 + 1.5, f"{v2:.1f}%", ha="center", fontsize=7.5)

    # Panel C: Image intensity & temperature distribution Lini A vs B
    idx_a = (tr_meta["lini"] == "A").values
    idx_b = (tr_meta["lini"] == "B").values
    mean_a = imgs[idx_a].mean(axis=(1, 2))
    mean_b = imgs[idx_b].mean(axis=(1, 2))
    axes[2].hist(mean_a, bins=40, alpha=0.65, density=True, label="Lini A (mean=139.8)", color="#1f77b4")
    axes[2].hist(mean_b, bins=40, alpha=0.65, density=True, label="Lini B (mean=111.7)", color="#d62728")
    axes[2].set_xlabel("Rata-rata Intensitas Piksel Citra")
    axes[2].set_ylabel("Densitas")
    axes[2].set_title("(c) Pergeseran Kecerahan Citra A vs B")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t1_eda_shift.png"), bbox_inches="tight")
    plt.close()
    print("Saved Task 1 EDA figures.")


def plot_task2_eda():
    rul_tr = pd.read_csv(os.path.join(DATA_DIR, "rul_train.csv.gz"))
    setel = rul_tr[["setel_1", "setel_2", "setel_3"]]
    setel_norm = (setel - setel.mean()) / setel.std()
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(setel_norm)
    rul_tr["regime"] = km.labels_
    max_cyc = rul_tr.groupby("id_unit")["siklus"].transform("max")
    rul_tr["rul"] = (max_cyc - rul_tr["siklus"]).clip(upper=150)

    s_cols = [f"s{i:02d}" for i in range(1, 22)]
    corrs = []
    for c in s_cols:
        med = rul_tr.groupby("regime")[c].transform("median")
        mad = rul_tr.groupby("regime")[c].transform(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
        z = ((rul_tr[c] - med) / (1.4826 * mad + 1e-8)).clip(-5, 5)
        corrs.append(z.corr(rul_tr["rul"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    # Panel A: Operating regimes scatter
    sample_idx = np.random.RandomState(42).choice(len(rul_tr), 6000, replace=False)
    sc = axes[0].scatter(rul_tr["setel_1"].iloc[sample_idx], rul_tr["setel_2"].iloc[sample_idx],
                         c=rul_tr["regime"].iloc[sample_idx], cmap="tab10", s=6, alpha=0.6)
    axes[0].set_xlabel("setel_1")
    axes[0].set_ylabel("setel_2")
    axes[0].set_title("(a) 6 Rezim Operasi Diskrit")

    # Panel B: Raw vs Regime-normalized s04 trajectory for 1 machine
    u1 = rul_tr[rul_tr["id_unit"] == rul_tr["id_unit"].iloc[0]].sort_values("siklus").copy()
    med_map = rul_tr.groupby("regime")["s04"].median()
    mad_map = rul_tr.groupby("regime")["s04"].apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
    u1["s04_norm"] = (u1["s04"] - u1["regime"].map(med_map)) / (1.4826 * u1["regime"].map(mad_map) + 1e-8)
    ax_twin = axes[1].twinx()
    l1 = axes[1].plot(u1["siklus"], u1["s04"], color="#999999", alpha=0.5, lw=0.8, label="s04 Mentah")
    l2 = ax_twin.plot(u1["siklus"], u1["s04_norm"].clip(-5, 5), color="#d62728", lw=1.2, label="s04 Norm. Rezim")
    axes[1].set_xlabel("Siklus (Unit 1)")
    axes[1].set_ylabel("s04 Mentah", color="#666666")
    ax_twin.set_ylabel("s04 Z-score Rezim", color="#d62728")
    axes[1].set_title("(b) Efek Normalisasi per Rezim (s04)")
    lines = l1 + l2
    axes[1].legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7)

    # Panel C: Sensor correlation with RUL
    colors = ["#2b5c8f" if abs(v) >= 0.05 else "#cccccc" for v in corrs]
    axes[2].bar(range(21), corrs, color=colors)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_xticks(range(21))
    axes[2].set_xticklabels(s_cols, rotation=60, fontsize=7)
    axes[2].set_ylabel("Korelasi Pearson dengan RUL")
    axes[2].set_title("(c) Informativitas 21 Sensor (7 Sensor Noise Abu-abu)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t2_eda_rul.png"), bbox_inches="tight")
    plt.close()
    print("Saved Task 2 EDA figure.")


def plot_task3_eda():
    cl_meta = pd.read_csv(os.path.join(DATA_DIR, "clustering_meta.csv"))
    cl_npz = np.load(os.path.join(DATA_DIR, "clustering.npz"))
    imgs = cl_npz["gambar"]
    spcs = cl_npz["spektrum"].astype(np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2))
    # Top row: 4 example microstructure images (2 from M1, 2 from M2)
    m1_idx = cl_meta[cl_meta["mikroskop"] == "M1"].index.values[:2]
    m2_idx = cl_meta[cl_meta["mikroskop"] == "M2"].index.values[:2]
    for i, (idx, m) in enumerate(zip(list(m1_idx) + list(m2_idx), ["M1", "M1", "M2", "M2"])):
        axes[0, i].imshow(imgs[idx], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"{cl_meta.loc[idx, 'id_sampel']} ({m}, mean={imgs[idx].mean():.0f})")
        axes[0, i].axis("off")

    # Bottom row: (0) Brightness M1 vs M2, (1-3) Raw vs despiked+baseline-corrected spectra
    idx_m1 = (cl_meta["mikroskop"] == "M1").values
    idx_m2 = (cl_meta["mikroskop"] == "M2").values
    axes[1, 0].hist(imgs[idx_m1].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, label="M1 (127.8)", color="#1f77b4")
    axes[1, 0].hist(imgs[idx_m2].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, label="M2 (146.1)", color="#ff7f0e")
    axes[1, 0].set_xlabel("Rata-rata Intensitas Citra")
    axes[1, 0].set_ylabel("Densitas")
    axes[1, 0].set_title("Perbedaan Kecerahan M1 vs M2")
    axes[1, 0].legend(fontsize=7)

    # Find 3 spectra with visible spikes/baseline
    diff2 = np.abs(spcs[:, 1:-1] - 0.5 * (spcs[:, :-2] + spcs[:, 2:])).max(axis=1)
    spike_idxs = np.argsort(diff2)[::-1][:3]
    from scipy.ndimage import median_filter, minimum_filter, uniform_filter1d
    for j, sidx in enumerate(spike_idxs):
        raw = spcs[sidx]
        despiked = median_filter(raw, size=5)
        baseline = uniform_filter1d(minimum_filter(despiked, size=19), size=11)
        clean = despiked - baseline
        axes[1, j + 1].plot(raw, color="#d62728", alpha=0.55, lw=0.9, label="Mentah (spike+baseline)")
        axes[1, j + 1].plot(clean, color="#1f77b4", lw=1.1, label="Bersih (despike-base)")
        axes[1, j + 1].set_title(f"Spektrum {cl_meta.loc[sidx, 'id_sampel']}")
        axes[1, j + 1].set_xlabel("Kanal Spektrum")
        if j == 0:
            axes[1, j + 1].legend(fontsize=6.5)

    fig.suptitle("Tugas 3: Eksplorasi Distorsi Citra Mikrostruktur dan Spektrum Komposisi", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "t3_eda_distortions.png"), bbox_inches="tight")
    plt.close()
    print("Saved Task 3 EDA figure.")


if __name__ == "__main__":
    plot_task1_eda()
    plot_task2_eda()
    plot_task3_eda()
