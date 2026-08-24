# Metodologi Penghitungan Jumlah Daun Tumpang‑Tindih dengan U‑Net

> **Dokumen ini menjelaskan secara mendalam cara proyek *segmentasi‑unet* mendeteksi **dan menghitung** jumlah daun, termasuk daun yang saling tumpang‑tindih**, menggunakan pendekatan **U‑Net dual‑head** dan **Watershed**.

---

## 1. Ringkasan Pipeline

```
Input Image (RGB) ──► Pre‑processing (resize, normalisasi)
       │
       ▼
┌───────────────────────�n│  U‑Net Dual‑Head Model │
│  • Head 1: Semantic mask (bg / leaf)   │
│  • Head 2: Distance map (per‑instance)   │
└─────────────┬─────────────┘
              │
          ┌───▼─────┐   ┌─────▼─────┐
          │  Mask   │   │ Distance │
          │  (B,2)  │   │ (B,1)    │
          └─────┬───┘   └─────┬─────┘
                │            │
                ▼            ▼
        Argmax → Binary mask   └─► Squeeze → Predicted distance map
                │
                ▼
        Watershed (peak‑local‑max on distance map)
                │
                ▼
        Instance label map (0 = background, 1…N = daun)
                │
                ▼
          **Count = N = label.max()**
```

Setiap blok di atas dijelaskan secara detail pada bagian berikut.

---

## 2. Stage 1 – Binary Segmentation dengan U‑Net Standar

- **Model**: `unet/unet_model.py` → `UNet(n_channels=3, n_classes=2, bilinear=False)`.
- **Loss**: *Weighted BCE* + *Dice* (`weighted_bce_dice`).  Berat kelas dihitung secara invers‑frekuensi untuk mengatasi ketidakseimbangan background ≫ foreground.
- **Keluaran**: Logits 2‑channel → `mask = logits.argmax(dim=1)` menghasilkan **mask biner** (0 = background, 1 = daun).
- **Hasil** pada data validasi: Dice ≈ 0.967, IoU ≈ 0.936.

Mask biner ini cukup untuk **segmentasi kasar**, tetapi tidak dapat memisahkan daun yang bersentuhan.

---

## 3. Stage 2 – Watershed pada Mask Biner (Baseline)

1. **Distance Transform** pada mask biner (`cv2.distanceTransform`). Nilai tertinggi berada di pusat tiap blob daun.
2. **Peak detection** menghasilkan *markers* awal.
3. **Watershed** pada citra `-distance` memisahkan blob menjadi *catchment basins* → masing‑masing basin = satu instance.
4. **Hitung** jumlah daun: `n_instances = labels.max()`.

**Kelemahan**: distance transform dihitung dari mask biner yang sudah diprediksi, sehingga noise atau kebocoran pada mask menyebabkan **peak tidak akurat** dan dapat menghasilkan under‑ atau over‑segmentation.

---

## 4. Stage 3 – Dual‑Head U‑Net (Metode yang Di‑implementasikan)

### 4.1 Mengapa Dual‑Head?
- **Head 1** (semantic) tetap melakukan segmentasi biner yang sudah terbukti akurat.
- **Head 2** memprediksi **distance map per‑instance** secara langsung, sehingga *peak* berada pada posisi yang **dipelajari** oleh jaringan, bukan dihitung dari mask yang berpotensi noisy.
- Kedua head berbagi **feature backbone** (encoder + decoder) sehingga parameter tambahan sangat sedikit (satu conv 1×1).

### 4.2 Arsitektur `UNetDualHead`
```python
class UNetDualHead(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)
        self.outc_dist = nn.Conv2d(64, 1, kernel_size=1)  # head‑2
    def forward(self, x):
        # encoder … decoder … (sama seperti UNet)
        x = self.up4(... )          # (B, 64, H, W)
        mask_logits = self.outc(x)      # (B, n_classes, H, W)
        dist_map    = self.outc_dist(x)  # (B, 1, H, W)
        return mask_logits, dist_map
```
- **Encoder** dikunci (freeze) dan dimuat dari checkpoint `best_binary.pth` → memanfaatkan representasi fitur yang sudah ter‑latih.
- **Decoder + heads** dilatih kembali dengan **dual loss**.

### 4.3 Target Distance Map (Ground‑Truth)
Untuk tiap gambar training, dibuat distance map **per‑instance**:
1. Membaca mask ber‑warna (setiap warna = satu daun).
2. Untuk setiap warna (kecuali background) dihitung `cv2.distanceTransform`.
3. Semua transformasi digabung dengan `np.maximum` sehingga **puncak tertinggi** berada pada pusat masing‑masing daun.
4. Normalisasi 0‑1 → `distance_target`.

### 4.4 Dual Loss
```python
def dual_loss(mask_logits, dist_pred, mask_true, dist_true,
              weight_clamp=(0.3,3.0), dist_weight=0.5):
    # --- classification (weighted BCE + Dice) ---
    n_fg = (mask_true==1).sum().float(); n_bg = (mask_true==0).sum().float()
    total = n_fg + n_bg
    w_fg = (total/(2*n_fg+1e-6)).clamp(*weight_clamp)
    w_bg = (total/(2*n_bg+1e-6)).clamp(*weight_clamp)
    weight = torch.tensor([w_bg,w_fg], device=mask_logits.device)
    bce = F.cross_entropy(mask_logits, mask_true, weight=weight)
    prob = F.softmax(mask_logits,dim=1)[:,1]
    dice = dice_loss(prob, mask_true.float(), multiclass=False)
    cls = bce + dice
    # --- regression (MSE) on distance map ---
    reg = F.mse_loss(dist_pred.squeeze(1), dist_true)
    return cls + dist_weight*reg, cls, reg
```
- `dist_weight` mengendalikan kontribusi loss regression (biasanya 0.3‑0.7).  
- Training berjalan 80 epoch dengan **learning‑rate 5e‑6**, encoder **frozen**, dan **early‑stop** pada validation Dice.

### 4.5 Watershed Menggunakan Prediksi Distance
```python
def watershed_from_distance(binary_mask, pred_distance, min_distance=12):
    # peak detection pada distance map (norm‑to‑255)
    coords = peak_local_max((pred_distance*255).astype(np.uint8),
                            min_distance=min_distance, labels=binary_mask)
    markers = np.zeros_like(pred_distance, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords)+1)
    markers = ndi.label(markers)[0]
    # background marker
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary_mask*255, kernel, iterations=3)
    markers[sure_bg==0] = 0
    # watershed pada -distance
    labels = watershed(-pred_distance, markers, mask=binary_mask.astype(bool))
    return labels  # 0 = bg, 1..N = daun
```
- **Peak‑local‑max** pada *predicted* distance map memberikan **marker yang lebih stabil** dibandingkan menggunakan distance transform dari mask biner.
- Parameter `min_distance` mengontrol seberapa jauh dua leaf harus ber‑jarak untuk dianggap terpisah; dapat disesuaikan per dataset.

### 4.6 Counting (Penjumlahan Daun)
Setelah watershed menghasilkan **label map** (`labels`), jumlah daun dihitung dengan satu baris:
```python
n_leaves = labels.max()   # karena label dimulai dari 1, 0 = background
```
Hasil ini dapat disimpan bersama visualisasi overlay untuk verifikasi.

---

## 5. Proses Inference pada Gambar Baru
```python
result = predict_single_image('path/to/image.jpg', model, device)
print('Jumlah daun terdeteksi:', result['n_instances'])
# result berisi: binary_mask, distance_map, probability, instances (label map)
```
Fungsi `predict_single_image` melakukan semua langkah: pre‑process, forward pass (dual‑head), watershed, dan mengembalikan dictionary yang siap dipakai untuk visualisasi atau penyimpanan.

---

## 6. Evaluasi Metodologi
| Metode | Kelebihan | Catatan |
|--------|-----------|---------|
| **Binary + Watershed** (Stage 2) | Tanpa training tambahan | Tergantung kualitas mask → sering under‑segmentasi pada daun yang tumpang‑tindih |
| **Dual‑Head + Watershed** (Stage 3) | Distance map dipelajari, peak lebih akurat, toleransi noise tinggi | Memerlukan loss tambahan (MSE) dan sedikit lebih lama training, namun meningkatkan **Count‑MAE** secara signifikan |
| **Instance‑Embedding (future)** | Tanpa watershed, clustering berbasis embedding | Memerlukan arsitektur lebih kompleks (discriminative loss) |

---

## 7. Langkah Selanjutnya (Roadmap)
1. **Evaluasi instance‑level metrics** (mAP, PQ) menggunakan `pycocotools` untuk mengukur kualitas segmentasi instance secara kuantitatif.
2. **Hyper‑parameter tuning**: `dist_weight`, `min_distance`, learning‑rate, dan apakah **unfreeze encoder** dapat meningkatkan akurasi pada dataset yang lebih variatif.
3. **Eksperimen dengan attention / deep‑supervision** (Enhanced‑U‑Net) untuk melihat dampak pada distance map.
4. **Penggunaan model lain** (Mask R‑CNN, Detectron2) sebagai baseline perbandingan.
5. **Deploy**: Membuat wrapper CLI/streamlit yang memanggil `predict_single_image` sehingga user dapat upload foto dan langsung mendapatkan jumlah daun.

---

## 8. Ringkasan Akhir
- **U‑Net dual‑head** menghasilkan **semantic mask** dan **distance map** secara bersamaan.
- **Watershed** memanfaatkan **peak pada distance map** untuk memisahkan daun yang tumpang‑tindih.
- **Penghitungan** jumlah daun hanyalah menghitung label unik (`labels.max()`).
- Metode ini menggabungkan **kemampuan representasi visual deep learning** dengan **algoritma klasik segmentasi** untuk menghasilkan counting yang akurat bahkan pada kondisi tumpang‑tindih yang berat.

*Dokumen ini dapat dijadikan referensi bagi anggota tim yang ingin memahami, memodifikasi, atau memperluas pipeline counting daun pada proyek ini.*