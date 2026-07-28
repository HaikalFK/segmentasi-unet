# Graph Report - .  (2026-07-28)

## Corpus Check
- Corpus is ~3,450 words - fits in a single context window. You may not need a graph.

## Summary
- 73 nodes · 107 edges · 8 communities (7 shown, 1 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- UNet Building Blocks
- Project Context & Docs
- Data Loading
- Training & Evaluation
- Model Assembly & Hub
- Prediction Utilities
- Data Download

## God Nodes (most connected - your core abstractions)
1. `U-Net Semantic Segmentation with PyTorch` - 12 edges
2. `BasicDataset` - 10 edges
3. `UNet` - 9 edges
4. `DoubleConv` - 7 edges
5. `evaluate()` - 6 edges
6. `train_model()` - 5 edges
7. `Down` - 5 edges
8. `Up` - 5 edges
9. `CarvanaDataset` - 5 edges
10. `dice_coeff()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Docker Image Publishing Workflow` --references--> `U-Net Semantic Segmentation with PyTorch`  [INFERRED]
  .github/workflows/main.yml → README.md
- `Python Package Dependencies (matplotlib, numpy, Pillow, tqdm, wandb)` --references--> `U-Net Semantic Segmentation with PyTorch`  [INFERRED]
  requirements.txt → README.md
- `train_model()` --calls--> `BasicDataset`  [EXTRACTED]
  train.py → utils/data_loading.py
- `train_model()` --calls--> `CarvanaDataset`  [EXTRACTED]
  train.py → utils/data_loading.py
- `Docker Image Publishing Workflow` --conceptually_related_to--> `Docker Container Runtime`  [INFERRED]
  .github/workflows/main.yml → README.md

## Import Cycles
- None detected.

## Communities (8 total, 1 thin omitted)

### Community 0 - "UNet Building Blocks"
Cohesion: 0.16
Nodes (8): DoubleConv, Down, OutConv, Parts of the U-Net model, Downscaling with maxpool then double conv, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 1 - "Project Context & Docs"
Cohesion: 0.18
Nodes (13): Docker Image Publishing Workflow, Biomedical Image Segmentation, Kaggle Carvana Image Masking Challenge, Sorensen-Dice Coefficient, Docker Container Runtime, Mixed Precision Training (AMP/FP16), Pretrained Carvana Model via torch.hub, PyTorch 1.13+ Deep Learning Framework (+5 more)

### Community 2 - "Data Loading"
Cohesion: 0.24
Nodes (6): Dataset, predict_img(), BasicDataset, CarvanaDataset, load_image(), unique_mask_values()

### Community 3 - "Training & Evaluation"
Cohesion: 0.40
Nodes (7): evaluate(), inference_mode, Tensor, train_model(), dice_coeff(), dice_loss(), multiclass_dice_coeff()

### Community 4 - "Model Assembly & Hub"
Cohesion: 0.28
Nodes (4): UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-…, unet_carvana(), Full assembly of the parts to form the complete network, UNet

### Community 5 - "Prediction Utilities"
Cohesion: 0.33
Nodes (3): ndarray, mask_to_image(), plot_img_and_mask()

## Knowledge Gaps
- **8 isolated node(s):** `download_data.sh script`, `Kaggle Carvana Image Masking Challenge`, `Sorensen-Dice Coefficient`, `Weights and Biases (wandb)`, `PyTorch 1.13+ Deep Learning Framework` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNet` connect `Model Assembly & Hub` to `UNet Building Blocks`, `Training & Evaluation`, `Prediction Utilities`?**
  _High betweenness centrality (0.362) - this node is a cross-community bridge._
- **Why does `BasicDataset` connect `Data Loading` to `Training & Evaluation`, `Prediction Utilities`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `U-Net Semantic Segmentation with PyTorch` (e.g. with `Docker Image Publishing Workflow` and `Biomedical Image Segmentation`) actually correct?**
  _`U-Net Semantic Segmentation with PyTorch` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `download_data.sh script`, `Kaggle Carvana Image Masking Challenge`, `Sorensen-Dice Coefficient` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._