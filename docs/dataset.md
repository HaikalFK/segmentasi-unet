# Spesifikasi Dataset — Plant Leaf Instance Segmentation

Dokumen ini merangkum spesifikasi dan praktik terbaik untuk dataset yang digunakan dalam proyek "segmentasi daun".

## 1. Sumber & lisensi
- Sumber contoh yang digunakan: pillisiddharth/plant-phenotyping-dataset (referensi di data/download_dataset.py).


## 2. Struktur direktori (standar proyek)
- data/imgs/   → file citra input (RGB), ekstensi: .png/.jpg/.npy/.pt
- data/masks/  → file mask per-image, berwarna (RGB) — tiap warna mewakili satu instance daun
- checkpoints/ → model & metadata (projek menyimpan mask_values dalam checkpoint)

## 3. Format & karakteristik berkas
- Citra: RGB, resolusi bervariasi. Proyek default resize ke (256, 256) untuk pelatihan (lihat train.py target_size).
- Mask: palet warna RGB; tiap warna (pixel value) mewakili satu daun berbeda di dalam gambar. Background biasanya satu warna (mis. (0,0,0)).
- Anotasi bersifat per-instance melalui warna; tidak ada file JSON terpisah — semua informasi ada di gambar mask.

## 4. Konsistensi warna
- Warna dapat berulang antar-gambar, tetapi tidak dijamin merepresentasikan objek yang sama secara global.
- Praktik yang aman: lakukan mapping warna→instance ID per-gambar saat memproses dataset (lihat langkah konversi di bawah).

## 5. Konversi warna → target pelatihan
Langkah yang direkomendasikan per-gambar:
1. Baca mask RGB (tanpa anti‑alias). Jika ada alpha, buang channel alpha.
2. Temukan semua warna unik selain background → setiap warna adalah satu instance lokal.
3. Buat dua target turunan:
   - Semantic mask: binary (0: background, 1: any-leaf) atau multiclass (0: bg, 1..N: instance classes) tergantung strategi.
   - Instance ID map: nilai integer per-pixel yang merepresentasikan ID instance unik lokal (1..K). Simpan mapping warna→ID untuk debugging.
4. (Opsional) Hapus warna artefak/anti‑alias dengan morphological opening atau thresholding warna.

## 6. Target tambahan untuk memisah tumpang-tindih
Untuk memudahkan pemisahan objek tumpang-tindih (instance breakup) buat target tambahan:
- Distance transform per-instance: jarak euklidean dari tiap piksel ke batas instance (atau ke centroid). Digunakan sebagai regresi (float map).
- Boundary map: peta tepi (1 pada tepi antar-instance, 0 di lain tempat).
- Marker seeds: puncak lokal (local maxima) pada distance transform untuk watershed marker.

Output yang biasa digunakan untuk model berbasis U-Net standar:
- Channel 0..C-1: segmentation logits (semantic / per-class)
- Channel D: predicted distance map (float normalized)
Atau gunakan dual-head: satu head untuk mask (classification), satu head untuk distance/boundary (regression).

## 7. Preprocessing yang direkomendasikan
- Resize ke target_size (project default: 256×256) menggunakan:
  - citra: bicubic / bilinear
  - mask: nearest (untuk menjaga nilai warna)
- Normalisasi citra (0..1 atau mean/std) sebelum masukkan ke model.
- Pastikan mask tetap integer/label setelah resize — gunakan nearest interpolation.
- Simpan dataset.mask_values (daftar warna yang dipindai) bersama checkpoint untuk reproduksibilitas.

## 8. Augmentasi yang direkomendasikan
- Geometrik: random crop, flip, rotation, scale, elastic deformations (paling membantu memisah instance yang tumpang-tindih)
- Photometric: brightness/contrast, hue/saturation jitter (untuk robust terhadap pencahayaan)
- Augmentasi harus diterapkan identik ke citra dan mask (transfer warna/label)

## 9. Loss & strategi pelatihan
- Semantic head: Cross-Entropy (multiclass) atau BCEWithLogits (binary) + Dice loss untuk per‑pixel overlap.
- Distance/boundary head: L1 atau MSE loss (regresi) — gabungkan dengan bobot yang dapat di-tuning.
- Weighted loss: gunakan weighting untuk menekankan foreground atau tepi jika imbalance.

## 10. Post-processing (memisah instance)
Contoh pipeline pasca-prediksi:
1. Ambil predicted semantic mask (thresholding jika probabilitas).
2. Ambil predicted distance map → cari local maxima (scipy.ndimage.maximum_filter) — jadikan marker.
3. Terapkan watershed pada inverted distance map dengan marker yang dihasilkan, dibatasi oleh semantic mask.
4. Bersihkan hasil (remove tiny regions, relabel kontigu secara konsisten).

Metode ini bekerja baik jika model memprediksi distance/boundary dengan cukup akurat.

## 11. Evaluasi & metrik
- Per-pixel: Dice (F1) dan mIoU (mean Intersection over Union).
- Instance-level: mAP at IoU thresholds (0.5..0.95), Panoptic Quality (PQ), Precision/Recall per-instance, F1-instance.
- Untuk tugas memisah daun tumpang-tindih, tambahkan metrik instance (PQ/AP) agar mengukur pemisahan objek bukan hanya overlap pixel.

## 12. Praktik engineering & metadata
- Simpan metadata per-experiment: mask color palette, preprocessing params (scale, target_size), augmentasi, random seed.
- Checkpoint yang disimpan oleh proyek sudah menyertakan mask_values — pertahankan kebiasaan ini.
- Untuk reproduksibilitas, simpan snapshot kecil dari beberapa mask & images sebagai sanity-check samples.

## 13. Catatan Colab / Notebook
- Unet_Binary_Leaf_Segmentation.ipynb dapat digunakan sebagai eksperimen interaktif:
  - Jalankan pipeline konversi warna→instance di notebook untuk verifikasi manual.
  - Jalankan training U-Net dengan dual‑head (semantic + distance) di Colab; Colab menyediakan GPU untuk percepatan.
  - Simpan artifacts (best model, metrics, sample predictions) ke Google Drive jika diperlukan.

## 14. Known issues / edge cases
- Anti-aliasing/blur pada mask dapat menghasilkan warna baru — gunakan nearest resize & per-pixel color snap/threshold.
- Jika suatu warna muncul sangat sedikit (singleton pixel), anggap sebagai noise dan filter dengan morphological opening.
- Jika palette tidak stabil antar gambar secara kuat, jangan gunakan label warna sebagai kelas global; selalu lakukan mapping per-gambar.

## 15. Referensi implementasi di repo
- utils/data_loading.py  — pembacaan gambar, scanning mask_values, preprocess
- train.py              — resize target_size, cara menyimpan mask_values di checkpoint, pelatihan U-Net
- docs/Unet_Binary_Leaf_Segmentation.ipynb (notebook) — contoh eksperimen interaktif

---
Jika ingin, dapat menambahkan contoh kode konversi warna→instance atau notebook template untuk dual-head U-Net; pilih apakah mau snippet Python di dokumen atau file util terpisah.
