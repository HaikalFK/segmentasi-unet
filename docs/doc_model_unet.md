# Dokumentasi Model U‑Net Standar

Dokumen ini merangkum arsitektur, implementasi, dan hasil pelatihan model U‑Net standar yang digunakan untuk segmentasi biner daun (leaf vs background) pada proyek **segmentasi‑unet**.

---

## 1. Ringkasan Arsitektur

U‑Net adalah jaringan **encoder‑decoder** dengan **skip connections** yang dirancang khusus untuk segmentasi citra. Struktur berbentuk huruf "U" terdiri dari dua jalur utama:

- **Encoder (contracting path)** – mengekstraksi fitur hierarkis dengan blok `Down` yang melakukan max‑pooling diikuti oleh dua konvolusi (`DoubleConv`).
- **Decoder (expanding path)** – mengembalikan resolusi spasial menggunakan up‑sampling (`Up`) dan menggabungkan kembali fitur dari encoder melalui *concatenation* pada resolusi yang sama.
- **Skip connections** – menjaga detail spasial tinggi yang hilang selama down‑sampling.
- **OutConv** – lapisan konvolusi 1×1 menghasilkan dua kanal output (background & leaf).

Gambar arsitektur dapat dilihat pada *graph* yang di‑export dengan **torchview** (lihat `docs/image-3.png`).

---

## 2. Implementasi dalam Kode

Berikut pemetaan komponen arsitektur ke file sumber:

| Komponen | File | Kelas / Fungsi |
|----------|------|----------------|
| **DoubleConv** – dua konvolusi 3×3 + BN + ReLU | `unet/unet_parts.py` | `class DoubleConv(nn.Module)` |
| **Down** – max‑pool + DoubleConv | `unet/unet_parts.py` | `class Down(nn.Module)` |
| **Up** – up‑sampling (bilinear atau transposed) + concat + DoubleConv | `unet/unet_parts.py` | `class Up(nn.Module)` |
| **OutConv** – konvolusi 1×1 | `unet/unet_parts.py` | `class OutConv(nn.Module)` |
| **UNet** – perakitan semua blok | `unet/unet_model.py` | `class UNet(nn.Module)` |

### 2.1. Detail Kelas Utama (`UNet`)

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)
```

- Parameter `bilinear` menentukan mode up‑sampling: **bilinear** (deterministik, sedikit parameter) atau **transposed convolution** (learnable, risiko *checkerboard*).
- `factor` menyesuaikan jumlah kanal pada bottleneck ketika bilinear dipilih, mengurangi memori.

---

## 3. Proses Pelatihan (Notebook `Unet_Binary_Leaf_Segmentation.ipynb`)

1. **Dataset** – `BinaryLeafDataset` mengubah mask multikelas menjadi biner (`mask > 0 → 1`). Ukuran gambar standar 256×256.
2. **Loss** – kombinasi **Weighted BCE** (inverse‑frequency) + **Dice loss** (`combined_loss`). Ini mengatasi ketidakseimbangan kelas.
3. **Optimiser** – `RMSprop` dengan learning rate `1e-5`, scheduler `ReduceLROnPlateau` (monitor Dice).
4. **Training loop** – early stopping dengan `patience=15` dan `delta=0.005`. Model disimpan pada `checkpoints/best_binary.pth`.
5. **Evaluasi** – metrik IoU dan Dice dicek pada set validasi; hasil terbaik: **Dice = 0.9668**, **IoU ≈ 0.94**.
6. **Visualisasi** – kurva loss, Dice, dan IoU disimpan sebagai `binary_learning_curve.png`.

---

## 4. Visualisasi Arsitektur

Grafik arsitektur dihasilkan dengan **torchview** pada notebook (cell 30) dan disimpan sebagai `docs/image-3.png`:

```python
import torchview
model_graph = torchview.draw_graph(
    model,
    input_size=(1, 3, 256, 256),
    hide_module_functions=True,
    save_graph=False,
)
model_graph.visual_graph
```

Gambar ini memperlihatkan alur data melalui semua blok `DoubleConv`, `Down`, `Up`, dan `OutConv` beserta dimensi tensor pada setiap tahap.

---

## 5. Penerapan Model dalam Alur Proyek

- **Notebook** `Unet_Binary_Leaf_Segmentation.ipynb` memuat model, melatih, dan menghasilkan checkpoint.
- **Inference** – selanjutnya pada tahap 2 (Watershed) model dipanggil untuk menghasilkan mask biner yang menjadi input pada prosedur pemisahan daun tumpang‑tindih.
- **Export** – checkpoint `best_binary.pth` serta grafik `image-3.png` disertakan dalam laporan ilmiah.

---

## 6. Ringkasan Hasil

| Metode | IoU | Dice |
|--------|-----|------|
| Binary U‑Net (final) | 0.9358 | **0.9668** |

Model berhasil mencapai akurasi tinggi pada segmentasi biner daun, memberikan fondasi yang kuat untuk tahap selanjutnya (Watershed leaf separation) dan untuk pengembangan model U‑Net yang lebih kompleks.

---

