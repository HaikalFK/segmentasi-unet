# Metrik Kuantitatif untuk Evaluasi Segmentasi Daun

Dokumen ini merangkum metrik yang relevan untuk eksperimen segmentasi daun (semantic & instance), termasuk definisi, interpretasi, kelebihan/kekurangan, dan catatan implementasi.

----
## 1. Ringkasan rekomendasi
- Per-pixel (semantic): Dice (F1), IoU (Jaccard), Accuracy, Precision & Recall per-class.
- Instance-level (pemecahan daun tumpang-tindih): COCO-style AP (mAP@IoU), AP@0.5 (AP50), AP@0.75 (AP75), Panoptic Quality (PQ), serta Precision/Recall/F1 per-instance.
- Boundary / shape: Hausdorff distance, Average Symmetric Surface Distance (ASSD) — opsional untuk analisis tepi.

----
## 2. Notasi umum
- Pred: prediksi (biner atau ter-threshold dari probabilitas)
- GT: ground truth (biner untuk semantic / instance masks untuk instance eval)
- |A|: jumlah piksel di himpunan A
- IoU(A,B): |A ∩ B| / |A ∪ B|
- Dice(A,B): 2|A ∩ B| / (|A| + |B|)

----
## 3. Per-pixel metrics (semantic)
1) Dice Coefficient (F1)
- Formula: Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
- Rentang: [0, 1], 1 = sempurna
- Kegunaan: sensitif pada overlap; sering dipakai pada medical/segmentasi objek kecil.

2) Intersection over Union (IoU / Jaccard)
- Formula: IoU = |Pred ∩ GT| / |Pred ∪ GT|
- Rentang: [0, 1]
- Catatan: Dice dan IoU berkorelasi; Dice = 2*IoU/(IoU+1)

3) Accuracy, Precision, Recall
- Accuracy = (TP + TN) / (TP+TN+FP+FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Catatan: accuracy misleading bila class imbalance besar; prefer Dice/IoU.

4) Multiclass / per-class
- Hitung per-class (one-vs-rest) lalu laporkan mean (mIoU) atau weighted mean.

----
## 4. Instance-level metrics
1) Average Precision (AP) — COCO style
- Untuk setiap prediksi instance, cocokkan dengan GT instance berdasar IoU; pred dikatakan TP jika IoU >= threshold.
- AP: area under precision–recall curve untuk sebuah IoU threshold.
- mAP@IoU (COCO): rata-rata AP pada beberapa IoU thresholds (0.50:0.05:0.95). Menilai ketepatan segmentation dan localisation.
- AP50 / AP75: AP pada threshold 0.5 / 0.75.
- Implementasi: pycocotools (convert instance masks → RLE / polygons).

2) Panoptic Quality (PQ)
- Dipakai untuk penilaian sekaligus segmentasi & deteksi instance (panoptic segmentation).
- Formula: PQ = (sum IoU over matched pairs) / (|TP| + 0.5|FP| + 0.5|FN|)
  - SQ (segmentation quality) = mean IoU over matched pairs
  - RQ (recognition quality) = |TP| / (|TP| + 0.5|FP| + 0.5|FN|)
- Rentang: [0,1]
- Catatan: PQ menggabungkan kualitas segmentasi dan kemampuan menemukan instance.

3) Instance-wise Precision/Recall/F1
- Setelah menentukan pasangan pred↔GT (paling sering lewat IoU threshold), hitung precision/recall pada tingkat instance.

4) Matching rules & edge cases
- Pilih IoU threshold(s) dan kebijakan multipair (umumnya one-to-one matching by highest IoU).
- Pastikan postprocessing (watershed, remove tiny regions) menghasilkan label instance konsisten.

----
## 5. Boundary / shape metrics (opsional)
- Hausdorff distance: jarak maksimum dari titik boundary satu mask ke boundary mask lain (peka terhadap outlier).
- Average Symmetric Surface Distance (ASSD): rata-rata jarak simetris antara batas pred & GT.
- Berguna ketika kualitas bentuk/tepi penting.

----
## 6. Per-hasil dan agregasi laporan
- Per-image dan rata-rata dataset: hitung metrik per-image lalu laporkan mean ± std.
- Juga laporkan distribusi (histogram) metrik untuk melihat variasi performa.
- Untuk instance metrics, laporkan breakdown menurut object size (small/medium/large) bila relevan.

----
## 7. Praktis: cara menghitung (snippet sederhana)
1) Dice dan IoU (numpy)

```python
import numpy as np

def dice_np(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)

def iou_np(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)
```

2) Instance AP / mAP
- Gunakan pycocotools: konversi setiap instance mask → RLE, buat COCO results, dan panggil pycocotools.coco.COCOeval.
- Implementasi manual butuh: sort predictions by score, greedy matching by IoU, hitung PR curve.

3) PQ
- Gunakan panopticapi (implementasi resmi) atau ikuti rumus matching di paper.

----
## 8. Tips evaluasi untuk dataset daun yang tumpang-tindih
- Pastikan evaluasi instance dilakukan terhadap hasil post-processing (watershed atau instance head output), bukan semantic mask langsung.
- Ukuran objek kecil: gunakan metrik breakdown by area.
- Tumpang-tindih tinggi: gunakan mAP pada rentang IoU rendah→tinggi dan PQ untuk melihat trade-off deteksi vs segmentation.

----
## 9. Tools & library yang direkomendasikan
- pycocotools (AP/mAP, COCO eval)
- panopticapi (PQ)
- scikit-image (label, watershed, regionprops)
- scipy.ndimage (distance_transform, maximum_filter)
- numpy / torch: per-pixel metrics

----
## 10. Catatan implementasi di repo
- Evaluate pipeline currently uses `evaluate.py` (lihat repo) yang menghitung per-pixel Dice & IoU untuk validation set. Untuk instance metrics, tambahkan conversion dari instance labels → COCO/panoptic format dan gunakan pycocotools/panopticapi.

----
## 11. Langkah selanjutnya (opsional)
- Tambahkan notebook `docs/evaluate_instance_metrics.ipynb` yang: convert predictions → instance masks, jalankan watershed, hitung mAP & PQ, visualisasikan false positives/negatives.
- Tambahkan util `utils/eval_utils.py` berisi helper untuk konversi RLE dan per-image metric aggregation.
