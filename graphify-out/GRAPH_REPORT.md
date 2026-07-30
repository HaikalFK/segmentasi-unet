# Graph Report - .  (2026-07-30)

## Corpus Check
- Corpus is ~9,675 words - fits in a single context window. You may not need a graph.

## Summary
- 106 nodes · 158 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Class Detection & Dataset Pipeline
- U-Net Architecture (unet_parts)
- Dataset Download & Organization
- Evaluation & Training Pipeline
- Data Loading & Preprocessing
- Prediction & Inference
- Model Hub & Registration
- Download Scripts
- CI/CD Workflow
- Python Dependencies

## God Nodes (most connected - your core abstractions)
1. `BasicDataset` - 11 edges
2. `UNet` - 10 edges
3. `Training Guide` - 9 edges
4. `Training Changes Log` - 8 edges
5. `DoubleConv` - 7 edges
6. `evaluate()` - 6 edges
7. `Project README` - 6 edges
8. `Class Determination Analysis` - 6 edges
9. `download_dataset()` - 5 edges
10. `main()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Carvana Image Masking Challenge Dataset` --semantically_similar_to--> `Plant Phenotyping Dataset (347 images, 347 masks)`  [INFERRED] [semantically similar]
  README.md → docs/class-determination.md
- `Automatic Mixed Precision Training` --conceptually_related_to--> `Training Guide`  [INFERRED]
  README.md → docs/training-guide.md
- `train_model()` --calls--> `BasicDataset`  [EXTRACTED]
  train.py → utils/data_loading.py
- `Training Guide` --references--> `U-Net Architecture in PyTorch`  [INFERRED]
  docs/training-guide.md → README.md
- `WandB Offline Mode Configuration` --conceptually_related_to--> `Weights and Biases Integration`  [INFERRED]
  docs/perubahan_training.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Training Documentation for Plant Phenotyping Dataset** — docs_class_determination_document, docs_perubahan_training_document, docs_training_guide_document, docs_class_determination_plantphenotypingdataset, docs_class_determination_autoclassdetection, docs_class_determination_pixelvalue27 [INFERRED 0.95]
- **Automated Class Detection Data Flow** — docs_class_determination_uniquemaskvalues, docs_class_determination_basicdataset, docs_class_determination_maskremapping, docs_class_determination_autoclassdetection [EXTRACTED 1.00]
- **Training Pipeline Stability Fixes** — docs_perubahan_wandboffline, docs_perubahan_cpusafedataloader, docs_perubahan_targetsize, docs_perubahan_setuptoolsfix [EXTRACTED 1.00]

## Communities (14 total, 3 thin omitted)

### Community 0 - "Class Detection & Dataset Pipeline"
Cohesion: 0.15
Nodes (22): Automated Class Detection from Mask Files, BasicDataset Class with Automated Mask Scanning, Class Determination Analysis, Mask Pixel Value Remapping for CrossEntropyLoss, Anomalous Pixel Value 27 in ara2012_plant033, Plant Phenotyping Dataset (347 images, 347 masks), unique_mask_values Scanning Function, CPU-Compatible DataLoader Configuration (+14 more)

### Community 1 - "U-Net Architecture (unet_parts)"
Cohesion: 0.16
Nodes (8): DoubleConv, Down, OutConv, Parts of the U-Net model, Downscaling with maxpool then double conv, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 2 - "Dataset Download & Organization"
Cohesion: 0.22
Nodes (13): download_dataset(), _find_dataset_root(), is_dataset_ready(), main(), organize_dataset(), Download script for Plant Phenotyping Dataset from Kaggle.  This script downlo, Copy RGB images to data/imgs/ and label masks to data/masks/.      Images and, Verify that images and masks match. (+5 more)

### Community 3 - "Evaluation & Training Pipeline"
Cohesion: 0.32
Nodes (8): evaluate(), iou_score(), Compute IoU (Intersection over Union) for each class and mean IoU.      Args:, Tensor, train_model(), dice_coeff(), dice_loss(), multiclass_dice_coeff()

### Community 4 - "Data Loading & Preprocessing"
Cohesion: 0.24
Nodes (6): Dataset, predict_img(), BasicDataset, CarvanaDataset, load_image(), unique_mask_values()

### Community 5 - "Prediction & Inference"
Cohesion: 0.22
Nodes (7): ndarray, compute_iou(), get_output_filenames(), mask_to_image(), Generate output filenames in the output directory, Compute IoU for a single prediction, plot_img_and_mask()

### Community 6 - "Model Hub & Registration"
Cohesion: 0.28
Nodes (4): UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-ima, unet_carvana(), Full assembly of the parts to form the complete network, UNet

## Knowledge Gaps
- **6 isolated node(s):** `download_data.sh script`, `CI/CD Docker Publish Workflow`, `Python Dependencies`, `Pretrained Model via Torch Hub`, `Combined CrossEntropyLoss and Dice Loss` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNet` connect `Model Hub & Registration` to `U-Net Architecture (unet_parts)`, `Evaluation & Training Pipeline`, `Prediction & Inference`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `train_model()` connect `Evaluation & Training Pipeline` to `Dataset Download & Organization`, `Data Loading & Preprocessing`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `BasicDataset` connect `Data Loading & Preprocessing` to `Evaluation & Training Pipeline`, `Prediction & Inference`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Training Guide` (e.g. with `Automated Class Detection from Mask Files` and `BasicDataset Class with Automated Mask Scanning`) actually correct?**
  _`Training Guide` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Training Changes Log` (e.g. with `Automated Class Detection from Mask Files` and `Anomalous Pixel Value 27 in ara2012_plant033`) actually correct?**
  _`Training Changes Log` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `download_data.sh script`, `CI/CD Docker Publish Workflow`, `Python Dependencies` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Class Detection & Dataset Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.1471861471861472 - nodes in this community are weakly interconnected._