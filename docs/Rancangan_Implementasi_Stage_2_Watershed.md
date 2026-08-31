# Rancangan Implementasi Stage 2 --- Instance Separation Daun Overlap

## Integrasi Modular dengan Pipeline Stage 1 U-Net

**Repository acuan:** `HaikalFK/segmentasi-unet`\
**Tujuan:** menambahkan Stage 2 dari paper, yaitu **Distance Transform +
Marker-Controlled Watershed**, tanpa mengubah siklus training Stage 1
yang sudah berjalan.

------------------------------------------------------------------------

## 1. Tujuan Rancangan

Stage 1 pada repository sudah berfungsi sebagai **semantic
segmentation**:

``` text
Input Image
    ↓
Binary U-Net
    ↓
Binary Leaf Mask
    ↓
Foreground: daun
Background: bukan daun
```

Masalahnya, ketika beberapa daun saling bersentuhan/overlap, U-Net dapat
menghasilkan satu area foreground yang menyatu:

``` text
Leaf A + Leaf B + Leaf C
          ↓
     satu blob mask
```

Stage 2 bertugas mengubah blob tersebut menjadi instance daun:

``` text
Binary Leaf Mask
      ↓
Distance Transform
      ↓
Local Maxima / Leaf Centers
      ↓
Markers
      ↓
Marker-Controlled Watershed
      ↓
Individual Leaf Instances
```

Stage 2 **tidak melakukan training model baru**. Stage 2 merupakan
post-processing/inference pipeline yang menerima output probabilitas
atau binary mask dari model Stage 1.

------------------------------------------------------------------------

## 2. Prinsip Integrasi

Prinsip utama implementasi:

1.  **Jangan mengubah training loop Stage 1.**
2.  **Checkpoint Stage 1 tetap menjadi sumber model.**
3.  Stage 2 hanya menerima output Stage 1.
4.  Preprocessing gambar untuk inference harus sama dengan Stage 1.
5.  Binary mask hasil Stage 1 menjadi input utama Watershed.
6.  Ground-truth instance mask tidak digunakan untuk menghasilkan
    prediction.
7.  Ground-truth instance mask hanya digunakan pada tahap evaluasi.
8.  Seluruh komponen Stage 2 dibuat modular.
9.  Notebook tetap menjadi orchestration layer, bukan tempat seluruh
    algoritma ditulis.
10. Parameter Stage 2 harus dapat diubah tanpa memodifikasi class inti.

Dengan desain ini:

``` text
                STAGE 1
┌──────────────────────────────────────┐
│ Dataset                              │
│   ↓                                  │
│ Binary preprocessing                 │
│   ↓                                  │
│ U-Net training                       │
│   ↓                                  │
│ Best checkpoint                      │
│   ↓                                  │
│ Binary prediction                    │
└──────────────────┬───────────────────┘
                   │
                   │ output contract
                   ▼
                STAGE 2
┌──────────────────────────────────────┐
│ Binary prediction                    │
│   ↓                                  │
│ Distance Transform                   │
│   ↓                                  │
│ Marker extraction                    │
│   ↓                                  │
│ Marker-controlled Watershed           │
│   ↓                                  │
│ Instance mask                        │
│   ↓                                  │
│ Instance evaluation                  │
└──────────────────────────────────────┘
```

------------------------------------------------------------------------

## 3. Kontrak Data Antara Stage 1 dan Stage 2

Agar kedua stage kredibel dan kompatibel, gunakan kontrak data yang
eksplisit.

### 3.1 Output Stage 1

Stage 1 sebaiknya menghasilkan minimal:

``` text
probability_map
binary_mask
```

Dengan bentuk konseptual:

``` python
probability_map: float32
shape = [H, W]

binary_mask: uint8
shape = [H, W]
values = {0, 1}
```

Probability map:

``` text
0.00 → sangat mungkin background
1.00 → sangat mungkin leaf
```

Binary mask:

``` text
0 → background
1 → leaf
```

Stage 2 **lebih baik menggunakan probability map untuk menghasilkan
binary mask**, tetapi tetap menyediakan opsi menerima binary mask yang
sudah tersedia.

------------------------------------------------------------------------

## 4. Struktur Folder yang Disarankan

Karena Stage 2 terdiri dari beberapa proses dan akan digunakan kembali,
lebih baik dibuat folder tersendiri daripada menumpuk kode ke notebook.

Struktur yang direkomendasikan:

``` text
segmentasi-unet/
│
├── unet/
│   ├── unet_model.py
│   └── unet_parts.py
│
├── Enhanced-U-Net/
│   ├── enhancedunet_model.py
│   ├── enhancedunet_parts.py
│   └── ...
│
├── stage2/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── distance_transform.py
│   ├── markers.py
│   ├── watershed.py
│   ├── pipeline.py
│   ├── visualization.py
│   └── evaluation.py
│
├── notebooks/
│   ├── Unet_Binary_Leaf_Segmentation.ipynb
│   └── Stage2_Watershed_Leaf_Instance_Segmentation.ipynb
│
├── checkpoints/
│   └── ...
│
├── predictions/
│   ├── stage1/
│   └── stage2/
│
└── ...
```

### Mengapa folder baru lebih baik?

Stage 2 bukan satu fungsi kecil. Minimal terdapat:

-   loading checkpoint,
-   inference,
-   probability thresholding,
-   distance transform,
-   marker detection,
-   watershed,
-   filtering instance,
-   visualization,
-   instance evaluation.

Notebook hanya memanggil class/function tersebut.

------------------------------------------------------------------------

## 5. Modul Stage 2

### 5.1 `stage2/config.py`

Berisi seluruh parameter Stage 2.

Contoh:

``` python
from dataclasses import dataclass


@dataclass
class WatershedConfig:
    probability_threshold: float = 0.5
    min_distance: int = 10
    foreground_threshold: float = 0.5
    min_instance_area: int = 100
    compactness: float = 0.0
```

Prinsip:

``` text
Notebook
   ↓
WatershedConfig
   ↓
Stage2Pipeline
```

Tidak perlu mengubah kode algorithm setiap kali parameter berubah.

------------------------------------------------------------------------

## 6. `stage2/preprocessing.py`

Modul ini bertanggung jawab menjaga kompatibilitas dengan Stage 1.

Fungsi utama:

``` python
class Stage2Preprocessor:
    def preprocess_image(self, image):
        ...
```

Tujuan:

``` text
Raw image
   ↓
same preprocessing as Stage 1
   ↓
Tensor
   ↓
Stage 1 model
```

Jangan membuat preprocessing baru yang berbeda secara diam-diam.

Jika Stage 1 menggunakan:

-   RGB,
-   resize tertentu,
-   normalization tertentu,

Stage 2 inference harus menggunakan pipeline yang sama.

------------------------------------------------------------------------

## 7. `stage2/prediction.py`

Modul untuk menggunakan checkpoint Stage 1.

Contoh API:

``` python
class Stage1Predictor:

    def __init__(self, model, checkpoint_path, device):
        ...

    def predict_probability(self, image):
        ...

    def predict_binary(self, image, threshold=0.5):
        ...
```

Output:

``` python
probability = predictor.predict_probability(image)

binary_mask = predictor.predict_binary(
    image,
    threshold=0.5
)
```

### Catatan penting

Stage 2 **tidak boleh melatih ulang U-Net**.

Checkpoint yang digunakan harus merupakan checkpoint Stage 1 yang sudah
selesai dilatih.

------------------------------------------------------------------------

## 8. `stage2/distance_transform.py`

Distance Transform adalah inti pertama Stage 2.

Untuk setiap pixel foreground:

``` text
distance(x, y)
=
jarak terdekat ke background
```

Secara konseptual:

``` text
boundary
████████████████
██     4      ██
██    5 5     ██
██   6 7 6    ██
██    5 5     ██
██     4      ██
████████████████
boundary
```

Pixel di tengah daun memiliki nilai distance yang lebih tinggi.

Untuk daun yang saling overlap, area pusat setiap daun cenderung
membentuk peak yang berbeda.

API:

``` python
class DistanceTransformer:

    def transform(self, binary_mask):
        """
        Returns:
            distance_map
        """
        ...
```

Output:

``` python
distance_map: np.ndarray
shape = [H, W]
dtype = float
```

------------------------------------------------------------------------

## 9. `stage2/markers.py`

Distance map belum cukup.

Kita perlu mencari local maxima sebagai kandidat pusat instance.

Pipeline:

``` text
Distance Map
      ↓
Local Maxima
      ↓
Candidate Centers
      ↓
Connected Marker Regions
      ↓
Marker Labels
```

API:

``` python
class MarkerGenerator:

    def __init__(
        self,
        min_distance=10,
        threshold_rel=0.2
    ):
        ...

    def generate(self, distance_map, binary_mask):
        ...
```

Output:

``` python
markers
```

Contoh:

``` text
0 0 0 0 0 0
0 0 1 0 0 0
0 0 0 0 2 0
0 0 0 0 0 0
```

Artinya:

``` text
1 = kandidat Leaf A
2 = kandidat Leaf B
```

------------------------------------------------------------------------

## 10. Mengapa Marker Sangat Penting?

Watershed tanpa marker dapat mengalami over-segmentation.

Contoh:

``` text
Distance Map
      ↓
Watershed
      ↓
banyak region kecil
```

Marker-controlled Watershed memberikan prior:

``` text
"Mulai pertumbuhan region dari titik-titik ini."
```

Sehingga:

``` text
Marker A → Leaf A
Marker B → Leaf B
Marker C → Leaf C
```

------------------------------------------------------------------------

## 11. `stage2/watershed.py`

Modul ini melakukan marker-controlled watershed.

API:

``` python
class MarkerControlledWatershed:

    def segment(
        self,
        distance_map,
        markers,
        binary_mask
    ):
        ...
```

Output:

``` python
instance_mask
```

Dengan format:

``` text
0 = background
1 = leaf instance 1
2 = leaf instance 2
3 = leaf instance 3
...
N = leaf instance N
```

Berbeda dengan Stage 1:

``` text
Stage 1:

0 = background
1 = leaf
```

Stage 2:

``` text
0 = background
1 = leaf #1
2 = leaf #2
3 = leaf #3
...
```

Ini adalah perubahan semantik paling penting antara kedua stage.

------------------------------------------------------------------------

## 12. `stage2/pipeline.py`

Ini menjadi facade utama.

Tujuannya agar notebook tidak perlu mengetahui detail setiap algoritma.

Contoh:

``` python
class Stage2Pipeline:

    def __init__(
        self,
        predictor,
        distance_transformer,
        marker_generator,
        watershed,
        config
    ):
        ...

    def process(self, image):
        ...
```

Pipeline internal:

``` text
image
  ↓
Stage1Predictor
  ↓
probability
  ↓
binary mask
  ↓
DistanceTransformer
  ↓
distance map
  ↓
MarkerGenerator
  ↓
markers
  ↓
MarkerControlledWatershed
  ↓
instance mask
```

Notebook cukup:

``` python
result = pipeline.process(image)
```

------------------------------------------------------------------------

## 13. Result Object

Agar debugging mudah, jangan hanya mengembalikan instance mask.

Gunakan object/dictionary terstruktur.

Contoh:

``` python
result = {
    "image": image,
    "probability": probability_map,
    "binary_mask": binary_mask,
    "distance_map": distance_map,
    "markers": markers,
    "instance_mask": instance_mask,
}
```

Dengan ini setiap tahap dapat diperiksa.

------------------------------------------------------------------------

## 14. Visualisasi

`stage2/visualization.py`

Minimal menyediakan:

``` python
class Stage2Visualizer:

    def show_pipeline(self, result):
        ...
```

Visualisasi:

``` text
┌─────────────┬──────────────┬───────────────┐
│ Input       │ Probability  │ Binary Mask   │
├─────────────┼──────────────┼───────────────┤
│ Distance    │ Markers      │ Instance Mask │
└─────────────┴──────────────┴───────────────┘
```

Tambahkan overlay:

``` text
Original Image
      +
Instance Boundary
      +
Instance ID
```

Contoh:

``` text
Leaf 1
Leaf 2
Leaf 3
...
```

Ini penting untuk menunjukkan bahwa Stage 2 benar-benar memisahkan daun
yang overlap.

------------------------------------------------------------------------

## 15. Evaluasi Stage 2

`stage2/evaluation.py`

Evaluasi harus dibedakan dari Stage 1.

### Stage 1

Mengukur semantic segmentation:

``` text
Dice
IoU
Precision
Recall
```

### Stage 2

Mengukur instance separation:

``` text
instance count error
instance precision
instance recall
instance F1
IoU per instance
```

Jika ground truth instance tersedia, dapat ditambahkan metrik
instance-level yang lebih kuat seperti AP/PQ sesuai definisi eksperimen
yang dipilih.

Jangan menggunakan Dice/IoU semantic sebagai satu-satunya bukti bahwa
overlapping leaves berhasil dipisahkan.

------------------------------------------------------------------------

## 16. Ground Truth Stage 2

Ini bagian yang harus dijaga agar evaluasi tidak bocor.

Stage 2 prediction:

``` text
Stage 1 Prediction
      ↓
Distance Transform
      ↓
Markers
      ↓
Watershed
```

Ground truth hanya digunakan:

``` text
Ground Truth Instance Mask
             ↓
        Evaluation
```

Bukan:

``` text
Ground Truth
    ↓
Markers
    ↓
Watershed
```

karena itu akan menjadi data leakage.

------------------------------------------------------------------------

## 17. Siklus Notebook yang Direkomendasikan

Notebook Stage 2 sebaiknya sederhana.

### Cell 1 --- Imports

``` python
from stage2.config import WatershedConfig
from stage2.prediction import Stage1Predictor
from stage2.distance_transform import DistanceTransformer
from stage2.markers import MarkerGenerator
from stage2.watershed import MarkerControlledWatershed
from stage2.pipeline import Stage2Pipeline
from stage2.visualization import Stage2Visualizer
```

### Cell 2 --- Configuration

``` python
config = WatershedConfig(
    probability_threshold=0.5,
    min_distance=10,
    foreground_threshold=0.5,
    min_instance_area=100
)
```

### Cell 3 --- Load Stage 1 Model

``` python
predictor = Stage1Predictor(
    model=model,
    checkpoint_path="checkpoints/best_model.pth",
    device=device
)
```

Model harus sama dengan model yang digunakan pada Stage 1.

### Cell 4 --- Initialize Stage 2

``` python
distance_transformer = DistanceTransformer()

marker_generator = MarkerGenerator(
    min_distance=config.min_distance
)

watershed = MarkerControlledWatershed()

pipeline = Stage2Pipeline(
    predictor=predictor,
    distance_transformer=distance_transformer,
    marker_generator=marker_generator,
    watershed=watershed,
    config=config
)
```

### Cell 5 --- Run One Image

``` python
result = pipeline.process(image)
```

### Cell 6 --- Visualize

``` python
visualizer = Stage2Visualizer()

visualizer.show_pipeline(result)
```

### Cell 7 --- Batch Processing

``` python
results = []

for image in test_images:
    result = pipeline.process(image)
    results.append(result)
```

### Cell 8 --- Evaluation

``` python
metrics = evaluator.evaluate(
    predictions=results,
    ground_truth=ground_truth_instances
)

print(metrics)
```

------------------------------------------------------------------------

## 18. Jangan Memasukkan Seluruh Implementasi ke Notebook

Untuk Stage 1, notebook mungkin masih nyaman karena training model
merupakan eksperimen utama.

Stage 2 berbeda.

Jika seluruh implementasi dimasukkan ke notebook:

``` text
Cell 1
Cell 2
Cell 3
...
Cell 30
...
Cell 50
```

maka akan sulit:

-   debugging,
-   reuse,
-   versioning,
-   eksperimen parameter,
-   membandingkan model,
-   menjalankan batch inference.

Karena itu rekomendasi final:

> **Gunakan folder `stage2/` untuk seluruh logic Stage 2. Notebook hanya
> sebagai runner/orchestrator.**

------------------------------------------------------------------------

## 19. Dependency yang Dibutuhkan

Implementasi dapat menggunakan library scientific Python yang sudah
umum:

``` text
numpy
scipy
scikit-image
opencv-python
matplotlib
torch
Pillow
```

Komponen utama:

``` python
scipy.ndimage.distance_transform_edt
```

untuk Distance Transform.

Dan:

``` python
skimage.feature.peak_local_max
skimage.segmentation.watershed
```

untuk marker detection dan watershed.

Pastikan versi library dicatat di environment repository agar eksperimen
dapat direproduksi.

------------------------------------------------------------------------

## 20. Parameter Stage 2 yang Harus Dieksperimenkan

Jangan langsung menganggap satu parameter sebagai parameter terbaik.

Parameter penting:

### Probability threshold

``` text
0.3
0.4
0.5
0.6
0.7
```

### Minimum marker distance

Contoh:

``` text
5
10
15
20
```

### Relative peak threshold

Contoh:

``` text
0.1
0.2
0.3
0.4
```

### Minimum instance area

Contoh:

``` text
50
100
200
500
```

Parameter ini sebaiknya disimpan dalam config.

------------------------------------------------------------------------

## 21. Eksperimen Bertahap

Jangan langsung melakukan tuning semua parameter sekaligus.

Gunakan urutan:

``` text
Experiment A
Default Watershed
        ↓
Baseline Stage 2
```

Kemudian:

``` text
Experiment B
Probability Threshold
        ↓
best threshold
```

Kemudian:

``` text
Experiment C
Marker Distance
        ↓
best marker distance
```

Kemudian:

``` text
Experiment D
Peak Threshold
        ↓
best peak threshold
```

Kemudian:

``` text
Experiment E
Minimum Instance Area
        ↓
best instance configuration
```

Terakhir:

``` text
Final Stage 2 Configuration
```

------------------------------------------------------------------------

## 22. Eksperimen Utama Penelitian

Setelah pipeline stabil, lakukan minimal:

``` text
Experiment 1
Standard U-Net
       ↓
Watershed
```

dan:

``` text
Experiment 2
Enhanced U-Net
       ↓
Watershed
```

Dengan Stage 2 yang sama.

Secara konseptual:

``` text
                    SAME DATA
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
     Standard U-Net          Enhanced U-Net
          │                         │
          ▼                         ▼
    Binary Prediction        Binary Prediction
          │                         │
          └────────────┬────────────┘
                       │
                SAME WATERSHED
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Instance Result A        Instance Result B
          │                         │
          └────────────┬────────────┘
                       ▼
                   Comparison
```

Ini membuat perbandingan arsitektur lebih fair.

------------------------------------------------------------------------

## 23. Hal yang Tidak Boleh Diubah Sembarangan

Agar hasil Stage 1 dan Stage 2 tetap kredibel:

### Jangan mengubah:

-   dataset split,
-   image preprocessing,
-   mask preprocessing,
-   checkpoint Stage 1,
-   model architecture,
-   training configuration,

hanya karena Stage 2 ditambahkan.

Jika ada perubahan, dokumentasikan sebagai eksperimen baru.

------------------------------------------------------------------------

## 24. Output Directory

Disarankan:

``` text
predictions/
│
├── stage1/
│   ├── probability/
│   └── binary/
│
└── stage2/
    ├── distance/
    ├── markers/
    ├── instances/
    └── overlays/
```

Dengan demikian hasil setiap tahap dapat diaudit.

------------------------------------------------------------------------

## 25. Format Penyimpanan Instance Mask

Jangan menyimpan instance mask sebagai RGB biasa jika akan digunakan
untuk evaluasi.

Gunakan integer label:

``` text
uint16
```

atau `int32` bila diperlukan.

Contoh:

``` text
0    background
1    leaf 1
2    leaf 2
3    leaf 3
...
```

Ini menjaga identitas instance.

Untuk visualisasi, barulah instance label dikonversi ke warna.

------------------------------------------------------------------------

## 26. Quality Control Sebelum Evaluasi

Sebelum menjalankan evaluasi massal, lakukan inspeksi:

``` text
Input
 ↓
Stage 1 binary
 ↓
Distance map
 ↓
Markers
 ↓
Watershed
```

Minimal cek 20--50 gambar secara visual, terutama:

-   daun tidak overlap,
-   dua daun touching,
-   dua daun overlap,
-   banyak daun overlap,
-   daun kecil,
-   daun dengan bentuk tidak beraturan,
-   foreground prediction yang noisy.

Tujuannya mengetahui jenis kegagalan Stage 2 sebelum tuning.

------------------------------------------------------------------------

## 27. Failure Modes yang Harus Dicatat

### Under-segmentation

Dua daun:

``` text
Leaf A + Leaf B
       ↓
     Leaf 1
```

Artinya marker terlalu sedikit atau distance map tidak menghasilkan peak
terpisah.

### Over-segmentation

Satu daun:

``` text
Leaf A
  ↓
Leaf 1 + Leaf 2 + Leaf 3
```

Artinya marker terlalu banyak.

### False instance

Noise:

``` text
noise blob
   ↓
instance
```

Dapat dikurangi dengan minimum instance area dan kualitas binary mask
Stage 1.

### Missing instance

Daun kecil tidak menghasilkan marker.

Ini harus diperhatikan ketika memilih `min_distance` dan threshold peak.

------------------------------------------------------------------------

## 28. Desain API Final

Target API yang sederhana:

``` python
pipeline = Stage2Pipeline.from_checkpoint(
    checkpoint_path="checkpoints/best_model.pth",
    config=config,
    device=device
)

result = pipeline.process(image)
```

Kemudian:

``` python
result.binary_mask
result.distance_map
result.markers
result.instance_mask
```

Jika ingin batch:

``` python
results = pipeline.process_batch(images)
```

Jika ingin menyimpan:

``` python
pipeline.save_results(
    result,
    output_dir="predictions/stage2/"
)
```

------------------------------------------------------------------------

## 29. Modularitas untuk Standard U-Net dan Enhanced U-Net

Stage 2 jangan mengetahui detail arsitektur model.

Hindari:

``` python
if model == "unet":
    ...
elif model == "enhanced_unet":
    ...
```

di dalam Watershed.

Sebaliknya gunakan interface:

``` text
Stage 1 Predictor
       ↓
probability map
       ↓
Stage 2
```

Sehingga:

``` text
Standard U-Net ──────┐
                     │
Enhanced U-Net ──────┼──→ Stage1Predictor contract
                     │
Model lain ──────────┘
                              ↓
                       Same Stage 2
```

Ini akan sangat berguna ketika penelitian berkembang.

------------------------------------------------------------------------

## 30. Urutan Implementasi yang Disarankan

Implementasikan secara bertahap:

``` text
STEP 1
Buat folder stage2/
```

↓

``` text
STEP 2
Buat DistanceTransformer
```

↓

``` text
STEP 3
Test Distance Transform secara standalone
```

↓

``` text
STEP 4
Buat MarkerGenerator
```

↓

``` text
STEP 5
Visualisasikan marker
```

↓

``` text
STEP 6
Buat MarkerControlledWatershed
```

↓

``` text
STEP 7
Test dengan binary ground truth
```

↓

``` text
STEP 8
Hubungkan output Stage 1
```

↓

``` text
STEP 9
Buat Stage2Pipeline
```

↓

``` text
STEP 10
Test end-to-end satu gambar
```

↓

``` text
STEP 11
Batch inference
```

↓

``` text
STEP 12
Instance evaluation
```

↓

``` text
STEP 13
Parameter tuning
```

↓

``` text
STEP 14
Standard U-Net vs Enhanced U-Net
```

------------------------------------------------------------------------

## 31. Unit Test Minimal

Sebelum menjalankan dataset penuh, buat synthetic mask:

``` text
   █████
 █████████
███████████
 █████████
   █████
```

untuk satu objek.

Expected:

``` text
number_of_instances = 1
```

Kemudian buat dua objek touching.

Expected:

``` text
number_of_instances = 2
```

Kemudian tiga objek.

Expected:

``` text
number_of_instances = 3
```

Ini memastikan algoritma Stage 2 benar sebelum dikaitkan dengan U-Net.

------------------------------------------------------------------------

## 32. Kriteria "Stage 2 Sudah Berhasil"

Jangan menyatakan Stage 2 berhasil hanya karena kode berjalan.

Minimal:

### Level 1 --- Pipeline

``` text
Stage 1 checkpoint
       ↓
binary mask
       ↓
distance transform
       ↓
markers
       ↓
watershed
```

semuanya berjalan tanpa error.

### Level 2 --- Visual

Daun touching/overlap dapat dipisahkan secara masuk akal.

### Level 3 --- Quantitative

Instance metrics menunjukkan bahwa separation benar-benar membaik.

### Level 4 --- Reproducibility

Parameter dan checkpoint dapat digunakan kembali untuk mendapatkan hasil
yang konsisten.

------------------------------------------------------------------------

## 33. Batasan Metodologis

Paper menjadi sumber metodologi utama untuk Stage 2. Implementasi kode
repository digunakan sebagai fondasi Stage 1.

Jangan mengklaim:

> "Repository telah mengimplementasikan paper."

Yang lebih tepat:

> "Repository menyediakan fondasi implementasi Stage 1 yang kompatibel
> dengan pendekatan paper. Stage 2 perlu ditambahkan sebagai
> post-processing instance separation menggunakan Distance Transform dan
> Marker-Controlled Watershed."

Ini lebih defensible secara akademik.

------------------------------------------------------------------------

## 34. Target Akhir

Setelah implementasi selesai, pipeline penelitian menjadi:

``` text
                         DATASET
                            │
                            ▼
                  Binary Mask Preparation
                            │
                            ▼
                    ┌───────────────┐
                    │   STAGE 1     │
                    │   U-NET       │
                    └───────┬───────┘
                            │
                     Best Checkpoint
                            │
                            ▼
                     Probability Map
                            │
                            ▼
                       Binary Mask
                            │
                            ▼
                    ┌───────────────┐
                    │   STAGE 2     │
                    │               │
                    │ Distance      │
                    │ Transform     │
                    │      ↓        │
                    │ Markers       │
                    │      ↓        │
                    │ Watershed     │
                    └───────┬───────┘
                            │
                            ▼
                  Instance Leaf Mask
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          Visualization             Evaluation
                │                       │
                ▼                       ▼
        Qualitative Result       Quantitative Result
```

------------------------------------------------------------------------

## 35. Rekomendasi Implementasi Final

**Gunakan folder baru `stage2/`.**

Jangan menaruh keseluruhan Stage 2 ke notebook.

Struktur minimal yang realistis:

``` text
stage2/
├── __init__.py
├── config.py
├── prediction.py
├── distance_transform.py
├── markers.py
├── watershed.py
├── pipeline.py
├── visualization.py
└── evaluation.py
```

Notebook:

``` text
Stage2_Watershed_Leaf_Instance_Segmentation.ipynb
```

hanya bertugas:

``` text
Configure
   ↓
Load checkpoint
   ↓
Initialize pipeline
   ↓
Run
   ↓
Visualize
   ↓
Evaluate
   ↓
Experiment
```

Dengan desain ini, **Stage 1 tetap tidak disentuh**, Stage 2 dapat diuji
secara independen, dan Standard U-Net maupun Enhanced U-Net dapat
menggunakan Stage 2 yang sama.

------------------------------------------------------------------------

## 36. Prinsip Akhir

Target desain bukan membuat kode sebanyak mungkin, tetapi membuat batas
antar-stage jelas:

``` text
STAGE 1
"Where are the leaves?"
        ↓
semantic foreground
        ↓
STAGE 2
"How many individual leaves are inside
this foreground?"
        ↓
instance segmentation
```

Stage 1 bertanggung jawab terhadap **foreground quality**.

Stage 2 bertanggung jawab terhadap **instance separation**.

Dengan pemisahan ini, kegagalan dan peningkatan masing-masing tahap
dapat dianalisis secara terpisah dan hasil penelitian menjadi jauh lebih
mudah dipertanggungjawabkan.
