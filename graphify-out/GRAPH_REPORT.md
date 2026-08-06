# Graph Report - .  (2026-08-04)

## Corpus Check
- 42 files · ~68,967 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 227 nodes · 359 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

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

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 20 edges
2. `UNet` - 15 edges
3. `BinaryLeafDataset` - 13 edges
4. `MinimalCheckpoint` - 12 edges
5. `BasicDataset` - 11 edges
6. `ComparisonCheckpoint` - 10 edges
7. `BinaryLeafDataset` - 9 edges
8. `WandBTracker` - 9 edges
9. `train_one_epoch()` - 8 edges
10. `validate()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `BinaryLeafDataset` --uses--> `UNet`  [INFERRED]
  experiments/compare_losses.py → unet/unet_model.py
- `BinaryLeafDataset` --uses--> `ComparisonCheckpoint`  [INFERRED]
  experiments/compare_losses.py → utils/checkpoint.py
- `BinaryLeafDataset` --uses--> `MinimalCheckpoint`  [INFERRED]
  experiments/compare_losses.py → utils/checkpoint.py
- `BinaryLeafDataset` --uses--> `WandBTracker`  [INFERRED]
  experiments/compare_losses.py → utils/wandb_tracker.py
- `create_model()` --calls--> `UNet`  [EXTRACTED]
  experiments/compare_losses.py → unet/unet_model.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (31): DataLoader, binary_metrics(), create_model(), create_optimizer(), create_scheduler(), load_config(), main(), _num() (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (15): deep_supervision_loss(), EnhancedUNet, Full assembly of the parts to form the complete Enhanced U-Net network Struktur…, Aktifkan gradient checkpointing. CATATAN: versi standar (unet/unet_model.py)…, Hitung loss untuk deep supervision. outputs : tuple dari EnhancedUNet.forward…, AttentionGate, DoubleConv, Down (+7 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (17): Any, ComparisonCheckpoint, MinimalCheckpoint, device, Module, Path, Minimal checkpoint manager for segmentation experiments. Design principle: Save…, List all saved checkpoints in this directory. (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (13): ndarray, compute_iou(), get_output_filenames(), mask_to_image(), predict_img(), Generate output filenames in the output directory, Compute IoU for a single prediction, BasicDataset (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (15): evaluate(), iou_score(), Compute IoU (Intersection over Union) for each class and mean IoU. Args: pred:…, inference_mode, train_model(), dice_coeff(), dice_loss(), multiclass_dice_coeff() (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (12): binary_metrics(), BinaryLeafDataset, combined_loss(), Dataset, device, Module, Tensor, train_2class.py — Binary Leaf Segmentation Training Train U-Net for binary… (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.16
Nodes (8): DoubleConv, Down, OutConv, Parts of the U-Net model, Downscaling with maxpool then double conv, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (9): check(), combined_loss(), ============================================================ TEST / CEK SETIAP…, Salinan loss dari notebook (BCE + Dice)., Tampilkan PASS/FAIL untuk satu pengujian., UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-…, unet_carvana(), Full assembly of the parts to form the complete network (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (13): download_dataset(), _find_dataset_root(), is_dataset_ready(), main(), organize_dataset(), Path, Download script for Plant Phenotyping Dataset from Kaggle. This script…, Copy RGB images to data/imgs/ and label masks to data/masks/. Images and masks… (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (7): Thin W&B integration for experiment tracking. Graceful degradation: if wandb is…, config = the `wandb:` section of YAML; loss_name = run tags/names., True if a wandb run is currently open and logging., Start a wandb run for this loss function., Log metrics to current wandb run. No-op if inactive., Wrap a CHW float tensor as a wandb.Image (or passthrough if no wandb)., WandBTracker

### Community 10 - "Community 10"
Cohesion: 0.32
Nodes (3): BinaryLeafDataset, Dataset, Binary leaf segmentation dataset (leaf=1, background=0).

## Knowledge Gaps
- **1 isolated node(s):** `download_data.sh script`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNet` connect `Community 7` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 10`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Why does `BinaryLeafDataset` connect `Community 10` to `Community 0`, `Community 9`, `Community 2`, `Community 7`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `MinimalCheckpoint` connect `Community 2` to `Community 0`, `Community 10`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `BinaryLeafDataset` (e.g. with `UNet` and `ComparisonCheckpoint`) actually correct?**
  _`BinaryLeafDataset` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `download_data.sh script` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12096774193548387 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09032258064516129 - nodes in this community are weakly interconnected._