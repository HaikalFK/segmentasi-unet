# Graph Report - .  (2026-08-21)

## Corpus Check
- 52 files · ~151,145 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 339 nodes · 495 edges · 29 communities (19 shown, 10 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 1% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Experiment Orchestration & Tracking
- Training Pipeline & Evaluation
- Enhanced U-Net Architecture
- Binary Segmentation Strategy
- Model Definitions & Dataset
- UNet Architecture Documentation
- Dataset & Training Configuration
- Checkpoint Management
- 2-Class Training Script
- UNet Building Blocks
- Dataset Download & Setup
- Prediction & Inference
- Segmentation Results Visualization
- File Explorer Tutorial Image
- ARA2012 Plant Sample Image
- U-Net Architecture Diagram
- Classification Report Image
- Training Log Screenshot
- Dataset Classes Documentation
- Loss Function Alternatives
- Data Download Script
- CI/CD Workflow
- ARA2012 Plant01 Image
- Enhanced U-Net Diagram
- Learning Rate Scheduler
- RMSprop Optimizer
- Test Time Augmentation
- Requirements Document

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 20 edges
2. `UNet` - 15 edges
3. `BinaryLeafDataset` - 13 edges
4. `MinimalCheckpoint` - 12 edges
5. `BasicDataset` - 11 edges
6. `ComparisonCheckpoint` - 10 edges
7. `BinaryLeafDataset` - 9 edges
8. `WandBTracker` - 9 edges
9. `Training Guide` - 9 edges
10. `train_one_epoch()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Carvana Image Masking Challenge Dataset` --semantically_similar_to--> `Plant Phenotyping Dataset (347 images, 347 masks)`  [INFERRED] [semantically similar]
  README.md → docs/class-determination.md
- `BinaryLeafDataset` --uses--> `ComparisonCheckpoint`  [INFERRED]
  experiments/compare_losses.py → utils/checkpoint.py
- `BinaryLeafDataset` --uses--> `MinimalCheckpoint`  [INFERRED]
  experiments/compare_losses.py → utils/checkpoint.py
- `BinaryLeafDataset` --uses--> `WandBTracker`  [INFERRED]
  experiments/compare_losses.py → utils/wandb_tracker.py
- `Automatic Mixed Precision Training` --conceptually_related_to--> `Training Guide`  [INFERRED]
  README.md → docs/training-guide.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Binary Segmentation + Watershed Pipeline Flow** — docs_binary_segmentation_pipeline_binaryleafdataset, docs_binary_segmentation_pipeline_unet, docs_binary_segmentation_pipeline_combined_loss [INFERRED 0.95]
- **Weighted BCE + Dice Loss Design** — docs_binary_segmentation_pipeline_background_collapse, docs_binary_segmentation_pipeline_inverse_frequency_weighting, docs_binary_segmentation_pipeline_combined_loss [INFERRED 0.95]
- **Automated Class Detection Data Flow** — docs_class_determination_uniquemaskvalues, docs_class_determination_basicdataset, docs_class_determination_maskremapping, docs_class_determination_autoclassdetection [EXTRACTED 1.00]
- **Training Pipeline Stability Fixes** — docs_perubahan_wandboffline, docs_perubahan_cpusafedataloader, docs_perubahan_targetsize, docs_perubahan_setuptoolsfix [EXTRACTED 1.00]
- **Training Documentation for Plant Phenotyping Dataset** — docs_class_determination_document, docs_perubahan_training_document, docs_training_guide_document, docs_class_determination_plantphenotypingdataset, docs_class_determination_autoclassdetection, docs_class_determination_pixelvalue27 [INFERRED 0.95]
- **Leaf Segmentation Pipeline** — docs_image_1_unet_architecture, docs_image_1_leaf_segmentation, docs_image_1_segmentation_results [INFERRED 0.75]
- **Windows File Explorer UI Layout** — docs_image_2_folder_navigation, docs_image_2_file_content_view, docs_image_2_tab_navigation, docs_image_2_file_action_buttons [EXTRACTED 1.00]
- **Segmentation Visualization Components** — docs_image-3_original_images, docs_image-3_ground_truth, docs_image-3_predicted_masks, docs_image-3_overlay [INFERRED 0.95]
- **Final Training Epoch Metrics** — docs_image4_training_log, docs_image4_training_loss_0072, docs_image4_validation_loss_00200, docs_image4_100_epochs [EXTRACTED 1.00]
- **Classification Metrics for Binary Leaf Segmentation** — docs_image_binary_leaf_segmentation, docs_image_class_0_background, docs_image_class_1_leaf [EXTRACTED 1.00]
- **U-Net Encoder-Decoder Assembly** — docs_penjelasan_model_unet_md_unet_model, docs_penjelasan_model_unet_md_doubleconv, docs_penjelasan_model_unet_md_down, docs_penjelasan_model_unet_md_up, docs_penjelasan_model_unet_md_outconv [EXTRACTED 1.00]
- **U-Net + Watershed Instance Segmentation Pipeline** — docs_tumpangtinding_md_instance_segmentation, docs_tumpangtinding_md_distance_transform, docs_tumpangtinding_md_watershed_postprocessing [EXTRACTED 0.95]
- **Enhanced U-Net Architecture Upgrades** — docs_penjelasan_model_unet_md_attention_gates, docs_penjelasan_model_unet_md_residual_doubleconv, docs_penjelasan_model_unet_md_deep_supervision [EXTRACTED 1.00]

## Communities (29 total, 10 thin omitted)

### Community 0 - "Experiment Orchestration & Tracking"
Cohesion: 0.06
Nodes (44): DataLoader, binary_metrics(), create_model(), create_optimizer(), create_scheduler(), load_config(), main(), _num() (+36 more)

### Community 1 - "Training Pipeline & Evaluation"
Cohesion: 0.11
Nodes (21): evaluate(), iou_score(), Compute IoU (Intersection over Union) for each class and mean IoU. Args: pred:…, inference_mode, predict_img(), train_model(), BasicDataset, CarvanaDataset (+13 more)

### Community 2 - "Enhanced U-Net Architecture"
Cohesion: 0.09
Nodes (15): deep_supervision_loss(), EnhancedUNet, Full assembly of the parts to form the complete Enhanced U-Net network Struktur…, Aktifkan gradient checkpointing. CATATAN: versi standar (unet/unet_model.py)…, Hitung loss untuk deep supervision. outputs : tuple dari EnhancedUNet.forward…, AttentionGate, DoubleConv, Down (+7 more)

### Community 3 - "Binary Segmentation Strategy"
Cohesion: 0.11
Nodes (25): Binary Leaf Segmentation Pipeline (U-Net), Mixed Precision Training (AMP), Background Collapse Problem, best_binary.pth Checkpoint, BinaryLeafDataset, Weighted BCE + Dice Combined Loss, Distance Transform (cv2.DIST_L2), Dice-Based Early Stopping & Model Selection (+17 more)

### Community 4 - "Model Definitions & Dataset"
Cohesion: 0.10
Nodes (12): check(), combined_loss(), ============================================================ TEST / CEK SETIAP…, Salinan loss dari notebook (BCE + Dice)., Tampilkan PASS/FAIL untuk satu pengujian., BinaryLeafDataset, Dataset, Binary leaf segmentation dataset (leaf=1, background=0). (+4 more)

### Community 5 - "UNet Architecture Documentation"
Cohesion: 0.11
Nodes (23): Automatic Mixed Precision (AMP), Attention Gates, Class Weighting (Inverse Frequency), Deep Supervision, Dice Loss + BCE Combined Loss, Dice Coefficient (F1 Score), DoubleConv, Down (Encoder Block) (+15 more)

### Community 6 - "Dataset & Training Configuration"
Cohesion: 0.15
Nodes (22): Automated Class Detection from Mask Files, BasicDataset Class with Automated Mask Scanning, Class Determination Analysis, Mask Pixel Value Remapping for CrossEntropyLoss, Anomalous Pixel Value 27 in ara2012_plant033, Plant Phenotyping Dataset (347 images, 347 masks), unique_mask_values Scanning Function, CPU-Compatible DataLoader Configuration (+14 more)

### Community 7 - "Checkpoint Management"
Cohesion: 0.16
Nodes (11): Any, MinimalCheckpoint, device, Module, Path, List all saved checkpoints in this directory., Delete all .pth files (keep configs & metrics)., Collect metrics from all loss experiments. (+3 more)

### Community 8 - "2-Class Training Script"
Cohesion: 0.14
Nodes (12): binary_metrics(), BinaryLeafDataset, combined_loss(), Dataset, device, Module, Tensor, train_2class.py — Binary Leaf Segmentation Training Train U-Net for binary… (+4 more)

### Community 9 - "UNet Building Blocks"
Cohesion: 0.16
Nodes (8): DoubleConv, Down, OutConv, Parts of the U-Net model, Downscaling with maxpool then double conv, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 10 - "Dataset Download & Setup"
Cohesion: 0.22
Nodes (13): download_dataset(), _find_dataset_root(), is_dataset_ready(), main(), organize_dataset(), Path, Download script for Plant Phenotyping Dataset from Kaggle. This script…, Copy RGB images to data/imgs/ and label masks to data/masks/. Images and masks… (+5 more)

### Community 11 - "Prediction & Inference"
Cohesion: 0.22
Nodes (7): ndarray, compute_iou(), get_output_filenames(), mask_to_image(), Generate output filenames in the output directory, Compute IoU for a single prediction, plot_img_and_mask()

### Community 12 - "Segmentation Results Visualization"
Cohesion: 0.73
Nodes (6): U-Net Binary Leaf Segmentation Documentation Image, Ground Truth Masks, Original Leaf Images, Overlay Visualization (Prediction + Ground Truth), Predicted Segmentation Masks, Segmentation Results Visualization

### Community 13 - "File Explorer Tutorial Image"
Cohesion: 0.40
Nodes (6): File Action Buttons (Back, New Folder), File Content View with Document Icons, Windows File Explorer Interface, Folder Navigation Panel, Tab Navigation (My Documents / Pictures), Cara Convert PDF ke Word (PDF to Word Conversion Tutorial)

### Community 14 - "ARA2012 Plant Sample Image"
Cohesion: 0.50
Nodes (5): ARA2012 Plant 03 Frame 04 RGB Image, Human Hand for Scale Reference, Binary Leaf Segmentation Training/Evaluation Sample, Tomato Plant Specimen (Indeterminate, Mature), White Growing Bucket

### Community 15 - "U-Net Architecture Diagram"
Cohesion: 0.80
Nodes (3): Binary Leaf Segmentation, Segmentation Results Visualization, U-Net Architecture Diagram

### Community 16 - "Classification Report Image"
Cohesion: 0.50
Nodes (5): Binary Leaf Segmentation Model Evaluation, Class 0 (Background), Class 1 (Leaf), Class Imbalance in Leaf Segmentation, Binary Segmentation Classification Report

### Community 17 - "Training Log Screenshot"
Cohesion: 0.50
Nodes (4): 100 Epochs Training Run, U-Net Training Log Output, Training Loss 0.0072, Validation Loss 0.0200

## Ambiguous Edges - Review These
- `image-1.png` → `Binary Leaf Segmentation`  [AMBIGUOUS]
  docs/image-1.png · relation: references
- `image-1.png` → `Segmentation Results Visualization`  [AMBIGUOUS]
  docs/image-1.png · relation: references
- `image-1.png` → `U-Net Architecture Diagram`  [AMBIGUOUS]
  docs/image-1.png · relation: references

## Knowledge Gaps
- **32 isolated node(s):** `download_data.sh script`, `CI/CD Docker Publish Workflow`, `Pretrained Model via Torch Hub`, `Combined CrossEntropyLoss and Dice Loss`, `RMSprop Optimizer with ReduceLROnPlateau Scheduler` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `image-1.png` and `Binary Leaf Segmentation`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `image-1.png` and `Segmentation Results Visualization`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `image-1.png` and `U-Net Architecture Diagram`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `UNet` connect `Model Definitions & Dataset` to `Experiment Orchestration & Tracking`, `Training Pipeline & Evaluation`, `2-Class Training Script`, `UNet Building Blocks`, `Prediction & Inference`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `BinaryLeafDataset` connect `Model Definitions & Dataset` to `Experiment Orchestration & Tracking`, `Checkpoint Management`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `MinimalCheckpoint` connect `Checkpoint Management` to `Experiment Orchestration & Tracking`, `Model Definitions & Dataset`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `BinaryLeafDataset` (e.g. with `UNet` and `ComparisonCheckpoint`) actually correct?**
  _`BinaryLeafDataset` has 4 INFERRED edges - model-reasoned connections that need verification._