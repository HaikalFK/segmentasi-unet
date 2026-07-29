# 📝 Catatan Perubahan — Training U-Net Plant Phenotyping

## 🔍 Validasi Data: Kenapa 21 Kelas? Kenapa Ada Nilai 27?

### Dataset

Dataset segmentasi daun (Plant Phenotyping) terdiri dari **347 gambar RGB** dan **347 mask label**.

Setelah memindai **semua 347 file mask**, ditemukan **22 nilai piksel unik**:

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27]
```

### Arti tiap angka

| Nilai Piksel | Makna           |
|-------------|-----------------|
| 0           | Background      |
| 1 – 20      | Daun ke-1 s.d. ke-20 |
| **27**      | **Label anomali** — hanya muncul di **1 file** (`ara2012_plant033.png`, 198 piksel dari total 112.530) |

### Kenapa Ada Nilai 27?

Nilai **27** adalah **label anomali/artifact** dari dataset asli. Kemungkinan penyebab:

1. **Kesalahan anotasi** — pemberi label secara tidak sengaja memakai nilai 27 (mungkin karena preset alat anotasi atau kesalahan teknis saat konversi)
2. **Noise konversi** — encoding yang tidak sempurna saat dataset dipublikasikan
3. **Label sengaja** — daun ke-20 mungkin ditandai 27 karena suatu alasan yang tidak tercatat di dokumentasi dataset

Yang jelas: **nilai 27 tidak berarti "kelas daun ke-27"** — secara botani, tanaman Arabidopsis yang difoto tidak memiliki 27 daun yang terlihat dalam satu frame.

### Kenapa Dokumentasi Bilang `--classes 21`?

Di dokumentasi resmi dataset, kelas segmentasi adalah **0–20** (21 kelas: 1 background + 20 daun). Nilai 27 dianggap **tidak seharusnya ada** — kemungkinan *artifact* dari proses anotasi.

Dulu kode **tidak bisa menangani** nilai 27 karena `CrossEntropyLoss` hanya menerima indeks kelas 0 hingga `n_classes-1`. Dengan argumen `--classes 21`:

- indeks kelas valid: **0 – 20**
- nilai piksel 27 dipetakan (di-*remap*) jadi indeks **21**
- indeks 21 > 20 → **error**

```
IndexError: Target 21 is out of bounds.
```

### Solusi: Deteksi Jumlah Kelas Otomatis

Sekarang kode **secara otomatis mendeteksi** jumlah kelas dengan memindai semua mask lebih dulu:

```
Scan semua 347 mask → kumpulkan semua nilai piksel unik → hitung jumlahnya → set n_classes
```

Hasil deteksi: **22 kelas** (nilai unik: 0, 1, 2, ..., 20, 27 → di-remap ke indeks 0, 1, 2, ..., 20, 21).

---

## ⚙️ Perubahan Kode

### 1. `requirements.txt`

| Sebelum | Sesudah | Alasan |
|---------|---------|--------|
| `wandb==0.13.5` | `wandb==0.19.1` | Versi lama butuh `pkg_resources` dari setuptools |
| *(tidak ada)* | `setuptools<60.0.0` | Setuptools ≥60 menghapus `pkg_resources` |

### 2. `train.py`

#### a. WandB → offline mode
```python
# Sebelum (butuh login)
experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
# Sesudah (jalan tanpa API key)
experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', mode='offline')
```

#### b. DataLoader → CPU-safe
```python
# Sebelum (error di CPU)
loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

# Sesudah (deteksi device)
num_workers = 0 if device.type == 'cpu' else os.cpu_count()
pin_memory = device.type == 'cuda'
```

#### c. Target size seragam
```python
# Dataset sekarang pakai ukuran tetap agar semua gambar bisa di-stack dalam batch
dataset = BasicDataset(..., target_size=(256, 256))
```

#### d. Deteksi kelas otomatis
```python
# Sebelum: jumlah kelas dari argumen CLI
model = UNet(n_channels=3, n_classes=args.classes, ...)

# Sesudah: jumlah kelas dari data
temp_dataset = BasicDataset(dir_img, dir_mask, args.scale, target_size=(256, 256))
n_classes = len(temp_dataset.mask_values)  # = 22
model = UNet(n_channels=3, n_classes=n_classes, ...)
```

### 3. `utils/data_loading.py`

- Baru: parameter `target_size` di `__init__`, `preprocess`, dan `CarvanaDataset`
- Baru: deteksi nilai piksel unik via `self.mask_values` (dari scan semua mask)

---

## 📊 Tabel Error & Solusi

| Error | Akar Masalah | Solusi |
|-------|-------------|--------|
| `ModuleNotFoundError: pkg_resources` | setuptools baru hapus `pkg_resources` | `wandb==0.19.1` + `setuptools<60` |
| `wandb auth token too short` (401) | API key tidak terdaftar | `mode='offline'` |
| `Trying to resize storage that is not resizable` | `pin_memory=True` di CPU | Nonaktifkan pin_memory saat CPU |
| `stack expects each tensor to be equal size` | Gambar dimensi berbeda-beda | `target_size=(256, 256)` |
| `Target 21 is out of bounds` | Mask punya nilai 27 → 22 kelas bukan 21 | Deteksi n_classes otomatis dari data |

---

## 🚀 Cara Menjalankan

```bash
python train.py --classes 21 --epochs 50 --batch-size 4 --scale 0.5
```

> **Catatan:** Argumen `--classes 21` tetap diterima untuk kompatibilitas, tetapi jumlah kelas aktual ditentukan oleh deteksi otomatis dari data (22 kelas, karena nilai piksel 27 ikut dihitung).