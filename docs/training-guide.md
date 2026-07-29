# Panduan Training U-Net — Segmentasi Plant Phenotyping Dataset

## 📋 Ringkasan

Panduan ini menjelaskan cara melatih (**training**) model U-Net untuk segmentasi daun tanaman menggunakan **Plant Phenotyping Dataset** yang sudah diunduh. Dataset terdiri dari **347 gambar RGB** dan **347 mask label** dengan **21 kelas segmentasi** (nilai piksel 0–20).

Arsitektur U-Net **tidak diubah** — yang diatur hanya parameter training (jumlah kelas, learning rate, batch size, dll.) melalui argumen baris perintah.

---

## 1️⃣ Prasyarat

### 1.1. Dataset sudah siap

Pastikan folder berikut sudah terisi:

```
data/
├── imgs/          ← 347 file .png (gambar RGB tanaman)
└── masks/         ← 347 file .png (mask segmentasi, nama sama persis dengan di imgs/)
```

> Jalankan `python data/download_dataset.py` jika belum. Script tersebut sudah punya guard — jika data sudah komplit, dia akan skip download.

### 1.2. Dependensi terinstall

Semua ada di `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Catatan:** PyTorch (`torch`, `torchvision`) perlu diinstall terpisah. Sesuaikan dengan CUDA versi GPU Anda:
>
> ```bash
> # Contoh untuk CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
>
> # Atau CPU-only
> pip install torch torchvision
> ```

### 1.3. Folder checkpoint (otomatis dibuat)

Hasil training akan disimpan ke folder `checkpoints/`. Folder ini dibuat otomatis saat training berjalan.

---

## 2️⃣ Konsep Penting: Jumlah Kelas

### Apa kata dokumentasi resmi

Dataset ini dipublikasikan memiliki **21 kelas** segmentasi:

| Nilai Piksel | Makna          |
|-------------|----------------|
| 0           | Background     |
| 1–20        | Daun/Label ke-1 s.d. ke-20 |

### Fakta dari data nyata

Setelah memindai **semua 347 file mask**, ditemukan **22 nilai piksel unik**:

```
[0, 1, 2, ..., 20, 27]
```

Angka **27** adalah **label anomali/artifact** — hanya muncul di **1 file** (`ara2012_plant033.png`, 198 piksel dari total 112.530). Kemungkinan kesalahan anotasi atau noise konversi dari dataset asli.

### Dampak ke training

**Jumlah kelas sekarang dideteksi otomatis dari data** (22 kelas, bukan 21). Model akan memetakan:

| Nilai mask asli | Indeks kelas |
|----------------|-------------|
| 0             | 0           |
| 1             | 1           |
| ...           | ...         |
| 20            | 20          |
| **27**        | **21**      |

> **Argumen `--classes` tetap bisa diberikan** (misal `--classes 21`) untuk kompatibilitas, tetapi kode akan **mengabaikannya** dan memakai hasil deteksi dari data.

Lihat `perubahan_training.md` untuk detail lengkap perubahan dan analisis data.

---

## 3️⃣ Menjalankan Training

Semua konfigurasi dilakukan melalui argumen baris perintah (`train.py`).

### 3.1. Training dasar (paling sederhana)

```bash
python train.py --classes 21 --epochs 50 --batch-size 4 --scale 0.5
```

### 3.2. Training dengan AMP (Mixed Precision) — **Direkomendasikan**

Akselerasi 1.5×–2× di GPU yang mendukung:

```bash
python train.py --classes 21 --epochs 50 --batch-size 4 --scale 0.5 --amp
```

### 3.3. Training dengan bilinear upsampling (lebih ringan)

Mengganti transposed convolution dengan bilinear upsampling — parameter berkurang, sedikit lebih cepat, kualitas hampir sama:

```bash
python train.py --classes 21 --epochs 50 --batch-size 8 --scale 0.5 --amp --bilinear
```

### 3.4. Training dengan validation 20%

Default validation split adalah 10%. Jika ingin lebih besar:

```bash
python train.py --classes 21 --epochs 50 --batch-size 4 --scale 0.5 --amp --validation 20
```

### 3.5. Melanjutkan training dari checkpoint

```bash
python train.py --classes 21 --epochs 100 --batch-size 4 --amp --load checkpoints/checkpoint_epoch50.pth
```

---

## 4️⃣ Penjelasan Argumen Training

| Argumen | Default | Fungsi |
|---------|---------|--------|
| `--epochs` / `-e` | 5 | Jumlah epoch. Untuk dataset 347 gambar, 50–100 epoch cukup. |
| `--batch-size` / `-b` | 1 | Batch size. Di GPU 4–8 GB: `4`. Di GPU 8+ GB: `8` atau `16`. |
| `--learning-rate` / `-l` | 1e-5 | Learning rate. Default sudah cocok untuk RMSprop. |
| `--scale` / `-s` | 0.5 | Skala resize. `0.5` = gambar diperkecil 50% — default terbaik. |
| `--validation` / `-v` | 10.0 | Persen data untuk validasi (0–100). |
| `--amp` | (off) | Mixed precision. Sangat disarankan di GPU modern. |
| `--bilinear` | (off) | Bilinear upsampling. Hemat memori, sedikit lebih cepat. |
| `--classes` / `-c` | 2 | **WAJIB diisi 21** untuk dataset ini. |
| `--load` / `-f` | (none) | Path ke checkpoint `.pth` untuk resume training. |

---

## 5️⃣ Apa yang Terjadi Saat Training Berjalan

1. **BasicDataset** akan memindai SEMUA file mask di `data/masks/` menggunakan multiprocessing untuk mendeteksi nilai piksel unik (0–20). Proses ini butuh beberapa detik dan menampilkan progress bar.
2. Dataset dibagi secara acak: **90% training** dan **10% validasi** (atau sesuai `--validation`).
3. **Loss function:** `CrossEntropyLoss` + `Dice Loss` (gabungan — lebih baik dari masing-masing sendiri).
4. **Optimizer:** RMSprop dengan `ReduceLROnPlateau` — learning rate otomatis turun jika validation Dice stagnan.
5. **Setiap beberapa langkah:**
   - Validasi Dice score dihitung
   - Learning rate scheduler menyesuaikan
   - Sample gambar + prediksi dikirim ke **Weights & Biases (wandb)**
6. **Setiap akhir epoch:** checkpoint disimpan ke `checkpoints/checkpoint_epoch{N}.pth`.

### 📊 Weights & Biases (WandB)

Training otomatis login ke wandb. Karena `anonymous='must'`, Anda bisa melihat log di terminal tanpa login. Jika ingin login:

```bash
wandb login
```

---

## 6️⃣ Evaluasi Model

### 6.1. Evaluasi otomatis (validation Dice)

`evaluate.py` sudah dipanggil otomatis selama training. Outputnya adalah **Dice score rata-rata** pada data validasi.

Untuk multiclass (21 kelas), Dice score dihitung **tanpa kelas background (kelas 0)** — jadi skor mencerminkan kualitas segmentasi daun saja.

### 6.2. Evaluasi manual dengan checkpoint tertentu

```bash
python evaluate.py --load checkpoints/checkpoint_epoch50.pth --classes 21 --scale 0.5
```

> **Catatan:** `evaluate.py` saat ini membaca argumen dari `train.py` melalui `get_args()`. Jika ada error, evaluasi langsung dari `train.py` dengan `--validation 100` bisa jadi alternatif.

---

## 7️⃣ Prediksi pada Gambar Baru

### 7.1. Prediksi satu gambar

```bash
python predict.py --model checkpoints/checkpoint_epoch50.pth \
                  --input path/to/gambar_baru.png \
                  --output hasil_prediksi.png \
                  --classes 21 --scale 0.5
```

### 7.2. Prediksi banyak gambar

```bash
python predict.py --model checkpoints/checkpoint_epoch50.pth \
                  --input data/imgs/ara2012_plant001.png data/imgs/ara2012_plant002.png \
                  --classes 21 --scale 0.5
```

Tanpa `--output`, nama file output otomatis: `{input_stem}_out.png`.

### 7.3. Lihat visualisasi langsung

```bash
python predict.py --model checkpoints/checkpoint_epoch50.pth \
                  --input data/imgs/ara2012_plant001.png \
                  --classes 21 --scale 0.5 --viz
```

### 7.4. Simpan tanpa visualisasi

```bash
python predict.py --model checkpoints/checkpoint_epoch50.pth \
                  --input data/imgs/ara2012_plant001.png \
                  --classes 21 --scale 0.5 --no-save
```

---

## 8️⃣ Rekomendasi Skenario Training

### 🟢 Pemula / CPU-only

```bash
python train.py --classes 21 --epochs 30 --batch-size 1 --scale 0.25
```

> `--scale 0.25` = gambar diperkecil hingga 25% agar muat di CPU.

### 🔵 GPU 4 GB (GTX 1650, RTX 3050)

```bash
python train.py --classes 21 --epochs 50 --batch-size 4 --scale 0.5 --amp
```

### 🟡 GPU 8 GB (RTX 3070, RTX 4060)

```bash
python train.py --classes 21 --epochs 75 --batch-size 8 --scale 0.5 --amp --bilinear
```

### 🟠 GPU 12+ GB (RTX 3080/4080/3090)

```bash
python train.py --classes 21 --epochs 100 --batch-size 16 --scale 1.0 --amp
```

> `--scale 1.0` = resolusi penuh (tanpa downscale).

---

## 9️⃣ Struktur Direktori (Hasil Training)

Setelah training selesai, struktur proyek akan seperti:

```
project-root/
├── data/
│   ├── imgs/                      ← Dataset gambar
│   ├── masks/                     ← Dataset mask
│   └── download_dataset.py
├── checkpoints/
│   ├── checkpoint_epoch10.pth     ← Checkpoint epoch 10
│   ├── checkpoint_epoch20.pth     ← Checkpoint epoch 20
│   ├── checkpoint_epoch30.pth     ← Checkpoint epoch 30
│   ├── checkpoint_epoch40.pth     ← Checkpoint epoch 40
│   └── checkpoint_epoch50.pth     ← Checkpoint epoch 50 (final)
├── docs/
│   └── training-guide.md          ← Dokumen ini
├── unet/
│   ├── unet_model.py              ← Arsitektur U-Net (tidak diubah)
│   ├── unet_parts.py              ← Blok pembangun U-Net
│   └── __init__.py
├── utils/
│   ├── data_loading.py            ← Dataset loader
│   ├── dice_score.py              ← Metrik evaluasi
│   └── utils.py
├── train.py                       ← Script training
├── predict.py                     ← Script prediksi
├── evaluate.py                    ← Script evaluasi
└── requirements.txt
```

---

## 🔟 Penyelesaian Masalah Umum

| Masalah | Solusi |
|---------|--------|
| **CUDA Out of Memory** | Turunkan `--batch-size` ke 1 atau 2, atau gunakan `--scale 0.25`. Kode sudah punya OOM recovery otomatis (gradient checkpointing). |
| **RuntimeError: CUDA error** | Pastikan driver GPU terupdate. Cek dengan `nvidia-smi`. |
| **Tidak ada GPU / CUDA** | Tambahkan `--device cpu` atau set `CUDA_VISIBLE_DEVICES=""`. |
| **WandB error / timeout** | Set `WANDB_MODE=disabled` untuk skip wandb: `WANDB_MODE=disabled python train.py ...` |
| **Dataset tidak ditemukan** | Pastikan `data/imgs/` dan `data/masks/` berisi file `.png` dan cocok. Jalankan `python data/download_dataset.py`. |
| **Loss NaN** | Turunkan `--learning-rate` (misal `-l 1e-6`). Atau tambahkan `--amp` yang kadang stabil. |
| **Dice score rendah** | Tambah epoch, turunkan learning rate, atau periksa apakah `--classes 21` sudah benar. |

---

## Referensi

- Dataset: [pillisiddharth/plant-phenotyping-dataset](https://www.kaggle.com/datasets/pillisiddharth/plant-phenotyping-dataset) (Kaggle)
- Arsitektur: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- Loss function: CrossEntropy + Dice Loss (gabungan)