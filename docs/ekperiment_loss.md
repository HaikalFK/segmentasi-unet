# Eksperimen Perbandingan Loss Function: Standard BCE vs Weighted BCE + Dice

## 1. Latar Belakang dan Tujuan

### 1.1. Konteks

Proyek ini menggunakan arsitektur U-Net untuk segmentasi daun tanaman (Plant Phenotyping Dataset). Dataset binary berisi **347 pasang gambar-mask** yang diubah menjadi segmentasi **2 kelas** (0 = background, 1 = daun), sehingga terdapat ketidakseimbangan kelas yang signifikan antara piksel background dan piksel daun.

### 1.2. Tujuan

Membuktikan secara **kuantitatif** bahwa kombinasi loss **Weighted BCE + Dice** lebih unggul daripada **Standard BCE** (baseline) untuk masalah segmentasi binary dengan class imbalance. Eksperimen ini berfokus pada **U-Net standar** (`unet/`).

### 1.3. Hipotesis

- **Standard BCE** pada data imbalanced akan menyebabkan model cenderung collapse memprediksi seluruh piksel sebagai background, sehingga Dice menuju 0.
- **Weighted BCE + Dice** akan menjaga keseimbangan antara ketepatan piksel (weighted CE) dan ketumpang-tindihan region (Dice), sehingga Dice jauh lebih tinggi.

---

## 2. Arsitektur Penerapan

Untuk menjaga modularitas dan reproduktibilitas, eksperimen ini dipisah ke beberapa modul khusus, tanpa mengubah base code U-Net.

```
segmentasi-unet/
├── unet/                        # Base code U-Net (TIDAK diubah)
│   ├── unet_parts.py
│   └── unet_model.py
├── utils/
│   ├── loss_functions.py        # Definisi semua loss function (modular)
│   ├── dice_score.py            # Dice coefficient + dice_loss (sudah ada)
│   ├── checkpoint.py            # Manajer checkpoint ringkas
│   └── wandb_tracker.py         # Integrasi W&B (graceful no-op)
├── experiments/
│   ├── compare_losses.py        # Orkestrator: train + bandingkan dua loss
│   └── configs/
│       ├── standard_bce.yaml    # Config baseline
│       └── weighted_bce_dice.yaml  # Config metode
└── docs/
    └── ekperiment_loss.md       # Dokumentasi ini
```

### 2.1. Prinsip Modular

Penempatan kode mengikuti satu prinsip: pemisahan tanggung jawab (separation of concerns).

| Modul | Tanggung Jawab |
|-------|----------------|
| `utils/loss_functions.py` | Mendefinisikan dan mendaftarkan loss function umum |
| `utils/dice_score.py` | Perhitungan Dice coefficient dan dice loss (dipakai eksperimen) |
| `utils/checkpoint.py` | Menyimpan hanya artefak penting (best model, config, metrics) |
| `utils/wandb_tracker.py` | Mencatat metrics/images ke W&B (tidak mengganggu alur utama) |
| `experiments/compare_losses.py` | Orkestrasi: pelatihan dua loss, penyimpanan, plot, ringkasan |
| `experiments/configs/*.yaml` | Hyperparameter identik antara dua loss (agar perbandingan adil) |

Perbandingan dilakukan dengan mengubah **satu-satunya variabel**: nama loss. Segala yang lain (data, split, seed, optimizer, scheduler, epoch) dijaga identik.

---

## 3. Definisi Loss Function

Didefinisikan dalam `utils/loss_functions.py`. Masing-masing loss diregistrasi ke `LOSS_REGISTRY` agar bisa dipilih melalui nama pada config.

### 3.1. Standard BCE

```python
def standard_bce(pred, target):
    return F.cross_entropy(pred, target)
```

- Menggunakan `CrossEntropyLoss` secara default, tanpa bobot kelas.
- Setiap piksel diperlakukan dengan bobot yang sama.
- Referensi ini menjadi **baseline** dalam eksperimen.

### 3.2. Weighted BCE + Dice

```python
def weighted_bce_dice(pred, target, weight_clamp=(0.3, 3.0), dice_weight=1.0):
    # 1) Bobot kelas frekuensi terbalik
    n_fg = (target == 1).sum().float()
    n_bg = (target == 0).sum().float()
    total = n_fg + n_bg
    w_fg = (total / (2 * n_fg + 1e-6)).clamp(*weight_clamp)
    w_bg = (total / (2 * n_bg + 1e-6)).clamp(*weight_clamp)
    weight = torch.tensor([w_bg, w_fg], device=pred.device)

    # 2) Weighted Cross-Entropy + Dice Loss pada foreground
    bce = F.cross_entropy(pred, target, weight=weight)
    prob = F.softmax(pred, dim=1)[:, 1]
    dice = dice_loss(prob, target.float(), multiclass=False)
    return bce + dice_weight * dice
```

**Langkah detail:**

1. **Hitung frekuensi kelas** pada batch: `n_fg` (piksel daun) dan `n_bg` (piksel background).
2. **Bobot frekuensi terbalik**: kelas yang jarang muncul (biasanya foreground) diberi bobot lebih besar, sehingga model tidak cenderung semata ke kelas mayoritas.
3. **Clamp** bobot ke rentang `[0.3, 3.0]` agar tidak ekstrem dan menjaga stabilitas.
4. **Weighted BCE**: diterapkan dengan bobot kelas ke `CrossEntropyLoss`.
5. **Dice Loss** dihitung pada probabilitas foreground (`softmax` channel 1), menggabungkan ketepatan piksel dengan kekayaan overlap.
6. **Total** = `weighted_bce + dice_weight * dice`.

Parameter yang dapat dikonfigurasi melalui YAML: `weight_clamp` (rentang clamp) dan `dice_weight` (kontribusi Dice).

---

## 4. Alur Eksekusi Eksperimen

Alur utama dijalankan oleh `experiments/compare_losses.py`:

```
Baca config YAML
      │
      ▼
Set seed (reproducible)
      │
      ▼
Load dataset binary (2 kelas) + split train/val (90/10, seed sama)
      │
      ▼
For tiap loss function (standard_bce, weighted_bce_dice):
      │
      ├── Buat model U-Net standar (2 kelas)
      ├── Siapkan optimizer + scheduler + AMP + loss
      ├── Inisialisasi W&B run (offline/online/disabled)
      │
      ├── Iterasi epoch:
      │       ├── train_one_epoch (dengan grad clipping)
      │       └── validate (IoU & Dice; val_loss dalam fp32)
      │
      ├── Early stopping berdasarkan best val_dice
      ├── Simpan best model + config + metrics (MinimalCheckpoint)
      └── Tutup W&B run
      │
      ▼
Bandingkan hasil kedua loss (ComparisonCheckpoint)
      │
      ├── metrics_comparison.csv
      ├── comparison_summary.md
      └── loss_comparison.png (+ kurva individual)
```

### 4.1. Reproduksibilitas

- `set_seed(seed)` menetapkan random seed untuk Python, NumPy, dan torch (CPU + CUDA) sebelum semua percobaan acak.
- `random_split` memakai generator seed yang identik untuk kedua eksperimen, sehingga **train/val split sama persis** antara standard_bce dan weighted_bce_dice. Ini syarat utama perbandingan yang adil.
- `cudnn.deterministic = True` → reproduksi deterministik di GPU.

---

## 5. Konfigurasi (Config YAML)

Setiap loss punya config sendiri. Berikut **kesamaan** yang wajib identik agar perbandingan fair:

| Kelompok | Parameter | Nilai |
|----------|-----------|-------|
| Model | `n_channels` | 3 |
| Model | `n_classes` | 2 |
| Model | `bilinear` | false |
| Training | `epochs` | 100 |
| Training | `batch_size` | 8 |
| Training | `learning_rate` | 1e-5 |
| Training | `optimizer` | RMSprop |
| Training | `weight_decay` | 1e-8 |
| Training | `momentum` | 0.999 |
| Training | `scheduler` | ReduceLROnPlateau |
| Training | `scheduler_patience` | 5 |
| Training | `scheduler_min_lr` | 1e-7 |
| Training | `amp` | true |
| Training | `early_stop_patience` | 15 |
| Training | `early_stop_delta` | 0.005 |
| Training | `gradient_clip` | 1.0 |
| Data | `target_size` | [256, 256] |
| Data | `validation_pct` | 10.0 |
| Data | `seed` (via experiment) | 42 |

Satu-satunya perbedaan adalah blok `loss`:

```yaml
# standard_bce.yaml
loss:
  name: "standard_bce"

# weighted_bce_dice.yaml
loss:
  name: "weighted_bce_dice"
  weight_clamp: [0.3, 3.0]
  dice_weight: 1.0
```

> **Catatan kritis optimasi:** Nilai `learning_rate: 1e-5` dipilih karena merupakan nilai yang telah terbukti stabil pada notebook binary (mencapai Dice ~0.96). LR yang lebih besar (mis. `1e-4`) menyebabkan pembias akibat RMSprop + momentum tinggi → model tidak terkontrol ke background.

---

## 6. Metrik dan Indikator Keberhasilan

Metrik dievaluasi pada **set validasi** (10% = 34 sampel) menggunakan fungsi `binary_metrics`:

- **IoU (Intersection over Union)** untuk foreground (kelas daun).
- **Dice coefficient** untuk foreground.

Kedua metrik dihitung sama persis antar kedua eksperimen untuk objektivitas.

### 6.1. Kriteria Penarikan Kesimpulan

- **Jika `standard_bce` menghasilkan** Dice mendekati 0 sementara **`weighted_bce_dice`** menghasilkan Dice signifikan (di atas 0.9), maka terbukti secara kuantitatif bahwa kombinasi loss lebih unggul.
- Data pendukung: `best_epoch`, `best_val_dice`, learning curve (train_loss, val_dice, val_iou) per epoch.

---

## 7. Manajemen Artefak (Hemat Resource)

Prinsip: **simpan minimal namun cukup untuk reproduksi + evaluasi.**

| Artefak | Ukuran kira-kira | Tujuan |
|---------|------------------|--------|
| `best_{loss}.pth` (state_dict) | ~124 MB per loss | Inference + evaluasi ulang |
| `config_{loss}.yaml` | ~1 KB | Reproduksibilitas penuh |
| `metrics_{loss}.json` | ~5 KB | Learning curve + aggregasi |
| `metrics_comparison.csv` | ~2 KB | Tabel perbandingan |
| `comparison_summary.md` | ~2 KB | Ringkasan markdown |
| `loss_comparison.png` (+ 3 plot) | ~200–500 KB | Visual learning curve |
| `wandb/offline-run-*` | Variabel | Logging eksternal (opsional sync) |

Tidak menyimpan checkpoint per epoch maupun state optimizer/scheduler, sehingga total resource jauh lebih kecil dari pendekatan menyimpan seluruh checkpoint.

---

## 8. Integrasi W&B (Opsional)

`utils/wandb_tracker.py` menyediakan integrasi W&B yang **tidak menggantikan artefak lokal**.

| Mode `wandb.mode` | Perilaku |
|-------------------|----------|
| `offline` | Menyimpan log lokal di folder `wandb/`, tidak diunggah otomatis. Bisa di-sync dengan `wandb sync` belakangan. |
| `online` | Langsung mengunggah ke cloud (perlu `wandb login`). |
| `disabled` | Menonaktifkan logging sama sekali. |

Properti penting:

- W&B dijalankan **per loss function** sebagai satu run dengan `group` yang sama (`bce-vs-weighted-bce-dice`), sehingga mudah **dibandingkan side-by-side** di dashboard.
- Setiap config bisa diubah ke `enabled: false` atau `mode: disabled` tanpa mengganggu jalannya eksperimen.

---

## 9. Cara Menjalankan

### 9.1. Jalankan dari CLI (local)

```bash
# Dua loss sekaligus
python experiments/compare_losses.py --config-dir experiments/configs --output-dir results

# Satu loss saja
python experiments/compare_losses.py --losses standard_bce --config-dir experiments/configs --output-dir results

# Buang file model setelah eksperimen (hemat disk, simpan config + metrics)
python experiments/compare_losses.py --config-dir experiments/configs --output-dir results --cleanup
```

### 9.2. Jalankan di Colab (via notebook binary `Unet_Binary_Leaf_Segmentation.ipynb`)

Section perbandingan (mark: cell [30]–[31]) menjalankan:

```bash
!python experiments/compare_losses.py --config-dir experiments/configs --output-dir results
```

flow:

1. Cell clone menggunakan branch `train_config` (konsisten dengan isi config perbandingan).
2. Cell [30] memastikan file perbandingan tersedia (fallback `origin/train_config`).
3. Cell [31] membaca `results/comparison/` dan menampilkan tabel lengkap serta plot learning curve.

---

## 10. Struktur Output Setelah Eksperimen Selesai

```
results/
├── standard_bce/
│   ├── best_standard_bce.pth
│   ├── config_standard_bce.yaml
│   └── metrics_standard_bce.json
├── weighted_bce_dice/
│   ├── best_weighted_bce_dice.pth
│   ├── config_weighted_bce_dice.yaml
│   └── metrics_weighted_bce_dice.json
└── comparison/
    ├── metrics_comparison.csv
    ├── comparison_summary.md
    ├── loss_comparison.png
    ├── loss_curves.png
    ├── dice_curves.png
    └── iou_curves.png
```

---

## 11. Catatan Teknis dan Pembelajaran

1. **Struktur dan alur eksperimen sudah benar** untuk tujuan membandingkan dua loss. Hasil training yang rendah pada pengujian pertama bukan karena arsitektur, melainkan karena setting konfigurasi (learning rate terlalu besar) dan bug closure pada loss. Keduanya telah diperbaiki.
2. **Bug closure recursion**: perhatikan bahwa variabel lokal `loss_fn` yang ditimpa lalu dipanggil di dalam closure akan merekursi ke dirinya sendiri. Solusinya: simpan fungsi asli di variabel terpisah (`base_loss_fn`).
3. **`val_loss` dihitung dalam fp32** di luar autocast untuk menghindari overflow fp16 yang menghasilkan NaN, sehingga angka loss tetap valid untuk perbandingan.
4. **W&B tidak wajib**: jika `mode: disabled` atau library `wandb` tidak tersedia, eksperimen tetap berjalan penuh dengan hasil lokal yang lengkap.

---

## 12. Rangkuman

- Proyek siap menjalankan **training U-Net standar** dan **membandingkan Standard BCE vs Weighted BCE + Dice** secara adil (split identik, hyperparameter identik, hanya loss yang berbeda).
- Kode modular (loss, checkpoint, tracking, config, orchestrator) sehingga mudah diperluas ke fungsi loss lain atau Enhanced U-Net.
- Keluarannya lengkap untuk analisa kuantitatif: tabel CSV, kurva belajar, best model `.pth`, dan metadata reproduksi.

Dokumentasi ini akan tetap relevan sebagai basis untuk pengembangan lebih lanjut (misalnya menambah loss baru atau memperluas ke Enhanced U-Net).