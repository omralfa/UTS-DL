# KONTRIBUSI ANGGOTA KELOMPOK

## 1. Identitas dan Pembagian Tugas

| No. | Nama | NIM | Kelas | Akun GitHub | Tanggung Jawab |
|---|---|---|---|---|---|
| 1 | Ruben Alfa Richie | 101032300008 | TK-47-06 | [@rubenalfarichie](https://github.com/rubenalfarichie) | Pembuat repository utama (`UTS-DL`), utilitas & EDA, **Task 1 (Klasifikasi Multimodal)**, dan **Task 2 (Regresi RUL)** |
| 2 | Omar Al Farizy Hutagalung | 101032300034 | TK-47-03 | [@omaralfa](https://github.com/omaralfa) | Fork & Pull Request, **Task 3 (Klastering Multimodal)**, dokumentasi (`kode/README.md`), dan penyusunan **laporan akhir (`laporan.pdf`)** |

## 2. Rincian Kontribusi

### Ruben Alfa Richie (`@rubenalfarichie`)
- Membuat dan mengelola repository utama kelompok (`https://github.com/rubenalfarichie/UTS-DL`), struktur pipeline reproducible (`kode/utils.py`, `kode/eda_figures.py`, `kode/run_all.py`, `kode/verify_submission.py`, `kode/requirements.txt`), serta melakukan review dan merge Pull Request.
- Mengerjakan **Task 1 — Klasifikasi Multimodal Cacat Produk** (`kode/task1_classification.py`, `weights/task1_cross_attn_seed42.pt`, `prediksi_klasifikasi.csv`): eksplorasi domain shift Lini A/B & missing sensor, perbandingan 5 arsitektur/fusi (Image-Only, Sensor-Only, Late Fusion, Early Concat, Gated Cross-Attention), audit mislabel & perbandingan loss robust (CE, Label Smoothing, Symmetric CE), uji missing modality, sanity check, dan ensemble 3-seed.
- Mengerjakan **Task 2 — Regresi Deret Waktu RUL** (`kode/task2_rul_regression.py`, `weights/task2_hybrid_seed42.pt`, `prediksi_rul.csv`): identifikasi 6 rezim operasi, seleksi 14 sensor informatif, normalisasi robust & imputasi kausal tanpa *future leakage*, perbandingan LightGBM baseline vs Dilated TCN vs BiGRU-Attention vs Hybrid CNN-BiGRU-Transformer, split ketat per mesin, estimasi ketidakpastian MC-Dropout + Conformal 90% PI, dan rekomendasi kebijakan servis.

### Omar Al Farizy Hutagalung (`@omaralfa`)
- Melakukan *fork* repository utama ke `https://github.com/omaralfa/UTS-DL`, bekerja pada branch fitur di fork pribadi, serta mengajukan Pull Request ke branch `main` repository utama.
- Mengerjakan **Task 3 — Klastering Multimodal Jenis Material** (`kode/task3_clustering.py`, `weights/task3_multimodal_ssl.pt`, `weights/task3_dec.pt`, `cluster.csv`): audit distorsi citra mikroskop M1/M2 & spektrum (spike/baseline/shift), pembelajaran representasi tanpa label (Multimodal SimCLR + Cross-Modal InfoNCE), penyelarasan domain per-mikroskop, analisis nilai dua modalitas, perbandingan K-Means, GMM, dan Deep Embedded Clustering (DEC), pemilihan K* = 6 dengan multi-kriteria, serta visualisasi t-SNE & profil klaster.
- Menyusun dan memfinalisasi laporan akhir (`laporan.pdf` via `kode/generate_report.py`), dokumentasi reproduksi (`kode/README.md`), serta dokumen `kontribusi.md`.

## 3. Riwayat Pull Request

Setiap anggota berkontribusi pada repository utama (`rubenalfarichie/UTS-DL`). Omar mengerjakan bagiannya melalui fork pribadi (`omaralfa/UTS-DL`) dan mengajukan Pull Request ke branch `main`:

| No. | Nama Anggota | Link Pull Request / Commit | Ringkasan | Status |
|---|---|---|---|---|
| 1 | Ruben Alfa Richie | https://github.com/rubenalfarichie/UTS-DL/commits/main | Inisialisasi repository utama, pipeline `kode/`, Task 1 (`prediksi_klasifikasi.csv`), dan Task 2 (`prediksi_rul.csv`) | Merged on `main` |
| 2 | Omar Al Farizy Hutagalung | https://github.com/rubenalfarichie/UTS-DL/pull/1 | Implementasi Task 3 Klastering Multimodal (`kode/task3_clustering.py`, `cluster.csv`, bobot SSL & DEC, grafik) | Merged *(sesuaikan nomor PR)* |
| 3 | Omar Al Farizy Hutagalung | https://github.com/rubenalfarichie/UTS-DL/pull/2 | Penyusunan laporan akhir (`laporan.pdf`, `kode/generate_report.py`), `kode/README.md`, dan `kontribusi.md` | Merged *(sesuaikan nomor PR)* |

## 4. Penggunaan Artificial Intelligence (AI)

Sesuai ketentuan UTS Deep Learning, berikut rincian penggunaan alat bantu AI oleh kelompok:

- **Alat AI yang digunakan**: Antigravity (Google Gemini).
- **Bagian yang dibantu**:
  - Membantu merancang implementasi PyTorch modular untuk arsitektur ResNet2D, ResNet1D-BiGRU, Gated Cross-Attention, Dilated TCN, Hybrid CNN-BiGRU-Transformer, Multimodal SimCLR + Cross-Modal InfoNCE, dan Deep Embedded Clustering (DEC).
  - Membantu menulis skrip visualisasi Matplotlib, skrip verifikasi otomatis (`kode/verify_submission.py`), serta skrip pembangkitan PDF ReportLab (`kode/generate_report.py`) berformat laporan akademik Indonesia.
  - Membantu analisis diagnostik terkait *covariate shift* antar lini produksi, normalisasi berbasis rezim operasi, dan penyelarasan embedding per-mikroskop.
- **Cara kelompok memverifikasi output**:
  - Seluruh skrip dijalankan ulang secara lokal menggunakan GPU NVIDIA RTX 3070 Ti pada dataset resmi UTS (`data/`) dengan random seed tetap (`42`, serta `42, 102, 202` untuk 3-seed).
  - Setiap metrik pada laporan dicocokkan langsung terhadap keluaran eksperimen (`results/task*_results.json`), dan seluruh berkas prediksi (`prediksi_klasifikasi.csv`, `prediksi_rul.csv`, `cluster.csv`) diverifikasi secara otomatis oleh `kode/verify_submission.py`.
  - Seluruh keputusan pemodelan dan hasil eksperimen dipahami oleh kedua anggota agar siap dijelaskan dan direproduksi langsung pada sesi *oral exam*.

## 5. Persetujuan Anggota Kelompok

Kami menyatakan bahwa pembagian tugas, kontribusi, dan rincian penggunaan AI dalam dokumen ini telah diketahui dan disetujui oleh seluruh anggota kelompok.

| Nama | NIM | Persetujuan |
|---|---|---|
| Ruben Alfa Richie | 101032300008 | Disetujui (24 November 2026) |
| Omar Al Farizy Hutagalung | 101032300034 | Disetujui (24 November 2026) |
