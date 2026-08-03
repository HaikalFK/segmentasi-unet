# Graph Report - .  (2026-07-31)

## Corpus Check
- Corpus is ~19,243 words - fits in a single context window. You may not need a graph.

## Summary
- 206 nodes · 299 edges · 16 communities (13 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.82)
- Token cost: 11,800 input · 5,200 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14

## God Nodes (most connected - your core abstractions)
1. `UNet` - 12 edges
2. `BasicDataset` - 11 edges
3. `BinaryLeafDataset` - 9 edges
4. `Training Guide` - 9 edges
5. `UNet Assembly Class` - 9 edges
6. `Training Changes Log` - 8 edges
7. `Standard U-Net (Ronneberger et al., MICCAI 2015)` - 8 edges
8. `DoubleConv` - 7 edges
9. `DoubleConv` - 7 edges
10. `evaluate()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Carvana Image Masking Challenge Dataset` --semantically_similar_to--> `Plant Phenotyping Dataset (347 images, 347 masks)`  [INFERRED] [semantically similar]
  README.md → docs/class-determination.md
- `Automatic Mixed Precision Training` --conceptually_related_to--> `Training Guide`  [INFERRED]
  README.md → docs/training-guide.md
- `combined_loss()` --calls--> `dice_loss()`  [EXTRACTED]
  train_2class.py → utils/dice_score.py
- `Training Guide` --references--> `U-Net Architecture in PyTorch`  [INFERRED]
  docs/training-guide.md → README.md
- `WandB Offline Mode Configuration` --conceptually_related_to--> `Weights and Biases Integration`  [INFERRED]
  docs/perubahan_training.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Automated Class Detection Data Flow** — docs_class_determination_uniquemaskvalues, docs_class_determination_basicdataset, docs_class_determination_maskremapping, docs_class_determination_autoclassdetection [EXTRACTED 1.00]
- **Training Pipeline Stability Fixes** — docs_perubahan_wandboffline, docs_perubahan_cpusafedataloader, docs_perubahan_targetsize, docs_perubahan_setuptoolsfix [EXTRACTED 1.00]
- **Training Documentation for Plant Phenotyping Dataset** — docs_class_determination_document, docs_perubahan_training_document, docs_training_guide_document, docs_class_determination_plantphenotypingdataset, docs_class_determination_autoclassdetection, docs_class_determination_pixelvalue27 [INFERRED 0.95]
- **Standard U-Net Building Blocks** — docs_penjelasan_model_une_unet, docs_penjelasan_model_une_doubleconv, docs_penjelasan_model_une_down, docs_penjelasan_model_une_up, docs_penjelasan_model_une_outconv [INFERRED 0.95]
- **Binary Segmentation + Watershed Pipeline Flow** — docs_binary_segmentation_pipeline_binaryleafdataset, docs_binary_segmentation_pipeline_unet, docs_binary_segmentation_pipeline_combined_loss, docs_binary_segmentation_pipeline_watershed_post_processing [INFERRED 0.95]
- **Weighted BCE + Dice Loss Design** — docs_binary_segmentation_pipeline_background_collapse, docs_binary_segmentation_pipeline_inverse_frequency_weighting, docs_binary_segmentation_pipeline_combined_loss, docs_penjelasan_model_une_combined_loss, docs_penjelasan_model_une_dice_loss [INFERRED 0.95]

## Communities (16 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.16
Nodes (14): evaluate(), iou_score(), Compute IoU (Intersection over Union) for each class and mean IoU.      Args:, predict_img(), train_model(), BasicDataset, CarvanaDataset, load_image() (+6 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (22): Binary Leaf Segmentation Pipeline (U-Net), Mixed Precision Training (AMP), Background Collapse Problem, BinaryLeafDataset, Weighted BCE + Dice Combined Loss, Distance Transform (cv2.DIST_L2), Gradient Health Checks (NaN/Inf skipping), Instance vs Semantic Segmentation (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (22): Automated Class Detection from Mask Files, BasicDataset Class with Automated Mask Scanning, Class Determination Analysis, Mask Pixel Value Remapping for CrossEntropyLoss, Anomalous Pixel Value 27 in ara2012_plant033, Plant Phenotyping Dataset (347 images, 347 masks), unique_mask_values Scanning Function, CPU-Compatible DataLoader Configuration (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (10): AttentionGate, DoubleConv, Down, OutConv, Parts of the Enhanced U-Net model  Versi ini adalah "U-Net standar" (lihat unet/, Upscaling then double conv (+ optional attention gate)      Struktur & argumen i, (convolution => [BN] => ReLU) * 2  (+ optional residual skip connection)      Sa, Downscaling with maxpool then double conv (+ optional dropout) (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (12): device, Module, binary_metrics(), BinaryLeafDataset, combined_loss(), Dataset, Tensor, train_2class.py — Binary Leaf Segmentation Training  Train U-Net for binary segm (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (8): DoubleConv, Down, OutConv, Parts of the U-Net model, Downscaling with maxpool then double conv, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (16): best_binary.pth Checkpoint, Dice-Based Early Stopping & Model Selection, Mixed Precision Training (AMP), Bilinear vs Transposed Conv Upsampling, Binary Metrics (IoU & Dice), Combined Loss (Weighted BCE + Dice), Dice Loss / dice_coeff, DoubleConv (+8 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (9): check(), combined_loss(), ============================================================  TEST / CEK SETIAP, Salinan loss dari notebook (BCE + Dice)., Tampilkan PASS/FAIL untuk satu pengujian., UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-ima, unet_carvana(), Full assembly of the parts to form the complete network (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (13): download_dataset(), _find_dataset_root(), is_dataset_ready(), main(), organize_dataset(), Download script for Plant Phenotyping Dataset from Kaggle.  This script downlo, Copy RGB images to data/imgs/ and label masks to data/masks/.      Images and, Verify that images and masks match. (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (7): ndarray, compute_iou(), get_output_filenames(), mask_to_image(), Generate output filenames in the output directory, Compute IoU for a single prediction, plot_img_and_mask()

### Community 10 - "Community 10"
Cohesion: 0.25
Nodes (5): deep_supervision_loss(), EnhancedUNet, Full assembly of the parts to form the complete Enhanced U-Net network  Struktur, Aktifkan gradient checkpointing.          CATATAN: versi standar (unet/unet_mode, Hitung loss untuk deep supervision.      outputs : tuple dari EnhancedUNet.forwa

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (8): Eksperimental Model U-Net Enhanced (empty), Penjelasan Model U-Net Standar (Standard U-Net Deep Dive), Attention Gate, Attention U-Net (Oktay et al., 2018), Enhanced U-Net, milesial/Pytorch-UNet Reference, Standard U-Net (Ronneberger et al., MICCAI 2015), U-Net++ (Zhou et al., 2018)

## Knowledge Gaps
- **14 isolated node(s):** `download_data.sh script`, `CI/CD Docker Publish Workflow`, `Pretrained Model via Torch Hub`, `Combined CrossEntropyLoss and Dice Loss`, `RMSprop Optimizer with ReduceLROnPlateau Scheduler` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNet` connect `Community 7` to `Community 0`, `Community 9`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `train_model()` connect `Community 0` to `Community 8`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Training Guide` (e.g. with `Automated Class Detection from Mask Files` and `BasicDataset Class with Automated Mask Scanning`) actually correct?**
  _`Training Guide` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `download_data.sh script`, `CI/CD Docker Publish Workflow`, `Pretrained Model via Torch Hub` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.1341991341991342 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.1471861471861472 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.1341991341991342 - nodes in this community are weakly interconnected._