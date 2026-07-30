# Binary Leaf Segmentation Pipeline — U-Net

**Project:** U-Net Segmentation for Plant Phenotyping Dataset  
**Notebook:** `Unet_Binary_Leaf_Segmentation.ipynb`  
**Model:** U-Net (2 class: background + leaf)  
**Post-Processing:** Watershed for overlapping leaf separation  
**Last Updated:** 2026-07-30

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 1: Binary Segmentation](#3-stage-1-binary-segmentation)
   - [3.1 Binary Dataset Wrapper](#31-binary-dataset-wrapper)
   - [3.2 Loss Function Design](#32-loss-function-design)
   - [3.3 Training Configuration](#33-training-configuration)
   - [3.4 Training Loop](#34-training-loop)
   - [3.5 Early Stopping & Model Selection](#35-early-stopping--model-selection)
4. [Stage 2: Leaf Separation (Watershed)](#4-stage-2-leaf-separation-watershed)
   - [4.1 Distance Transform](#41-distance-transform)
   - [4.2 Marker Generation](#42-marker-generation)
   - [4.3 Watershed Algorithm](#43-watershed-algorithm)
5. [Evaluation Protocol](#5-evaluation-protocol)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Results Interpretation](#7-results-interpretation)
8. [Common Issues & Troubleshooting](#8-common-issues--troubleshooting)
9. [Next Steps](#9-next-steps)

---

## 1. Overview

This pipeline performs **binary leaf segmentation** using the standard U-Net architecture, followed by **Watershed post-processing** to separate overlapping leaves into individual instances.

### Why Binary?

The Plant Phenotyping dataset uses instance segmentation labels (each leaf has a unique ID: 1, 2, 3, ...). However, these IDs are **not semantically consistent** across images — "leaf 1" in plant A looks completely different from "leaf 1" in plant B. A standard U-Net cannot learn these as distinct semantic classes.

Converting to binary (background = 0, leaf = 1) transforms the problem into a simpler **foreground vs. background** task that U-Net handles well.

### Two-Stage Approach

| Stage | Task | Output |
|-------|------|--------|
| **1** | Binary U-Net segmentation | Pixel-wise mask: 0 = background, 1 = leaf |
| **2** | Watershed separation | Instance labels: 0 = background, 1..N = individual leaves |

---

## 2. Pipeline Architecture

```
Input Image (RGB)
     │
     ▼
┌─────────────────────────────────┐
│  Binary Dataset Wrapper         │
│  • All pixels > 0 → 1 (leaf)   │
│  • Resize to 256×256           │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  U-Net (2 classes)             │
│  • Encoder: 4 downsampling     │
│  • Decoder: 4 upsampling       │
│  • Output: 2 channels (bg/leaf)│
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Loss: BCE (weighted) + Dice   │
│  • Inverse frequency weighting │
│  • Prevents background collapse│
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  argmax → Binary Mask (0/1)    │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Watershed Post-Processing     │
│  • Distance Transform          │
│  • Marker Generation           │
│  • Watershed Separation        │
└─────────────────────────────────┘
     │
     ▼
Instance Labels (1..N leaves)
```

---

## 3. Stage 1: Binary Segmentation

### 3.1 Binary Dataset Wrapper

**File:** `BinaryLeafDataset` class (inline in notebook)

The dataset wrapper converts multi-class instance masks to binary on-the-fly, without modifying the original mask files on disk.

**Algorithm:**

```
For each mask file:
  1. Read PNG as NumPy array (values: 0, 1, 2, ..., 27)
  2. Resize to 256×256 using NEAREST interpolation
     (preserves label integrity — no fractional pixels)
  3. Apply: binary = (mask > 0).astype(int64)
     • All leaf labels → 1
     • Background (0)   → 0
```

**Key properties:**

| Property | Value |
|----------|-------|
| Input channels | 3 (RGB) |
| Target size | 256 × 256 |
| Image resize | BICUBIC |
| Mask resize | NEAREST (no label interpolation) |
| Normalization | pixel / 255.0 |
| Output | `{"image": (3,256,256), "mask": (256,256), "name": str}` |

### 3.2 Loss Function Design

**Formula:**

```
Loss = BCE_weighted + Dice_loss
```

#### BCE (Weighted Cross-Entropy)

Standard BCE punishes errors on all pixels equally. With >99% background pixels, the model learns to predict "background" for every pixel — the **background collapse problem**.

**Solution:** Inverse frequency weighting.

```
For each batch:
  n_leaf = count of leaf pixels
  n_bg   = count of background pixels
  total  = n_leaf + n_bg

  w_leaf = total / (2 × n_leaf)     ← large weight (rare class)
  w_bg   = total / (2 × n_bg)       ← small weight (common class)
  weight = clamp(w, 0.3, 3.0)       ← numerical safety

  BCE = cross_entropy(pred, target, weight=[w_bg, w_leaf])
```

**Effect:** Misclassifying a leaf pixel costs ~10× more than misclassifying a background pixel. This forces the model to actively learn leaf boundaries.

#### Dice Loss

```
Dice = 2 × |pred ∩ true| / (|pred| + |true|)
Dice_loss = 1 - Dice
```

Dice loss directly optimizes the overlap between prediction and ground truth, making it robust to class imbalance.

**Combined effect:** BCE provides pixel-level gradient signal; Dice provides region-level overlap optimization. Together they produce smooth, complete leaf masks.

### 3.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 | Upper bound — early stopping triggers sooner |
| Batch size | 8 | Balances GPU memory and gradient stability |
| Learning rate | 1×10⁻⁵ | Stable for RMSprop with AMP (from 22-class experiments) |
| Optimizer | RMSprop (lr=1e-5, momentum=0.999, weight_decay=1e-8) | Matches original U-Net paper |
| Scheduler | ReduceLROnPlateau (patience=5, min_lr=1e-7) | Reduces LR when validation plateaus |
| AMP | True (with gradient scaling) | Mixed precision for faster training |
| Validation split | 10% | 313 train / 34 validation |
| Early stop patience | 15 epochs | Prevents overfitting |
| Improvement delta | 0.005 | Minimum Dice improvement to reset patience |

### 3.4 Training Loop

**Per-epoch algorithm:**

```
for each batch:
  1. Load images, move to GPU
  2. Forward pass (autocast if AMP enabled)
  3. Compute combined_loss(pred, target)
  4. Check for NaN/Inf → skip if present
  5. Backward pass (scaler.scale)
  6. Unscale gradients (scaler.unscale)
  7. Clip gradients to max norm 1.0
  8. Check gradient health → skip step if NaN/Inf
  9. Optimizer step (scaler.step)
  10. Update scaler (scaler.update)

After all batches:
  1. Validation pass (no gradients, no AMP)
  2. Compute mean IoU and Dice across validation set
  3. Update learning rate scheduler
  4. Check early stopping condition
```

**Gradient health checks:** AMP can produce NaN gradients from float16 overflow. The loop skips problematic batches rather than crashing, ensuring training continues.

### 3.5 Early Stopping & Model Selection

**Metric:** Validation Dice score (not IoU — Dice is more stable for small foreground regions).

**Logic:**

```
best_dice = -1.0   ← ensures epoch 1 is always saved

if avg_dice > best_dice + delta:
    best_dice = avg_dice
    save checkpoint('best_binary.pth')
    stale_epochs = 0
else:
    stale_epochs += 1

if stale_epochs >= patience:
    stop training
```

**Checkpoints saved:**

| File | Contents |
|------|----------|
| `checkpoints/best_binary.pth` | Model weights only (lowest validation loss) |
| `checkpoints/binary_unet_complete.pth` | Dict: `{"model": state_dict, "history": training_log}` |

---

## 4. Stage 2: Leaf Separation (Watershed)

The binary mask groups all leaves into a single foreground class. Watershed separates these into individual instances using the **distance transform** as a height map.

### 4.1 Distance Transform

```
Input:  Binary mask (255 = leaf, 0 = background)
        ↓
Distance Transform (cv2.DIST_L2)
        ↓
For each leaf pixel, compute Euclidean distance
to the nearest background pixel.
        ↓
Output: Float array where center of each leaf
has high values, edges have low values.
```

**Visual analogy:** Think of each leaf as a hill. The distance transform creates a mountain peak at the center of each leaf, with slopes down to the boundaries.

### 4.2 Marker Generation

Markers are the "seeds" that Watershed uses as starting points for each leaf.

```
Distance Map
    ↓
Normalize to [0, 255]
    ↓
Threshold at 40% of max
    ↓
Connected Components Analysis
    ↓
Each isolated region → one marker (seed point)
```

**Threshold tuning:** `0.4 × max_distance` works for well-separated leaves. For heavier overlap, reduce to `0.25 × max_distance` to get more markers.

### 4.3 Watershed Algorithm

```
Input: Grayscale image (distance map as intensity)
                            ↓
Markers: 1..N (1 per leaf seed), 0 = unknown, -1 = boundary
                            ↓
Watershed flood-fill from each marker
                            ↓
Each pixel assigned to the nearest marker's basin
                            ↓
Output: Label matrix where:
  0        = background
  1..N     = individual leaves
  -1       = watershed boundary (leaf-leaf border)
```

**When Watershed works well:**
- Leaves are connected but have distinct central regions
- Overlap is moderate (< 50% leaf area overlapping)

**When Watershed struggles:**
- Extreme overlap (one leaf almost completely covers another)
- Small or thin leaves with no clear center
- Binary mask errors (missing leaf parts)

---

## 5. Evaluation Protocol

### Binary Segmentation Metrics

| Metric | Formula | Range | Target |
|--------|---------|-------|--------|
| **IoU** (Intersection over Union) | `TP / (TP + FP + FN)` | [0, 1] | > 0.85 |
| **Dice** (F1) | `2TP / (2TP + FP + FN)` | [0, 1] | > 0.90 |

Where:
- TP = pixels correctly predicted as leaf
- FP = background pixels incorrectly predicted as leaf
- FN = leaf pixels incorrectly predicted as background

### Final Evaluation Procedure

1. Load best model weights (`checkpoints/best_binary.pth`)
2. Run inference on entire validation set (34 images)
3. Compute mean IoU and Dice across all predictions
4. Visualize 4 random samples for qualitative inspection

### Watershed Evaluation (Qualitative)

- Count number of detected leaves vs. visual inspection
- Check for over-segmentation (one leaf split into multiple)
- Check for under-segmentation (multiple leaves merged)

---

## 6. Key Design Decisions

### Decision 1: Binary over Multi-Class

| Aspect | Binary (2 class) | Multi-Class (22 class) |
|--------|-----------------|----------------------|
| Class imbalance | ~1:100 (leaf:bg) | ~1:20 each class |
| Model capability | ✅ U-Net standard | ❌ Needs instance seg |
| Training stability | ✅ Stable | ❌ mIoU ~0.009 |
| Interpretability | ✅ Clear | ❌ Arbitrary class IDs |

### Decision 2: Weighted BCE + Dice

Choosing the right loss function is critical for imbalanced segmentation:

| Loss | Pros | Cons |
|------|------|------|
| BCE only | Simple, fast | Collapses to background |
| Dice only | Handles imbalance | Unstable early training |
| BCE + Dice (weighted) | ✅ Best of both | Slightly more computation |

### Decision 3: Gradient Health Checks

AMP (Automatic Mixed Precision) speeds training by using float16 where safe. However, softmax + CrossEntropy in float16 can overflow to NaN.

The fix: **check for NaN before backward**, not after. This avoids the `unscale_()` double-call error while keeping AMP benefits.

### Decision 4: Early Stopping on Dice

IoU is the final metric of interest, but Dice is more stable during training for small foreground regions. Early stopping uses Dice; final reporting shows both.

---

## 7. Results Interpretation

### Expected Learning Curve

```
Epoch 1:  Dice ~ 0.30   — Model finds rough leaf regions
Epoch 2:  Dice ~ 0.40   — Boundaries improve
Epoch 3:  Dice ~ 0.60   — Most leaves detected
Epoch 4-5: Dice ~ 0.85  — High quality masks
Epoch 6-12: Dice ~ 0.96 — Near-perfect binary segmentation
```

### Metric Meaning

| Dice Score | Interpretation |
|-----------|---------------|
| < 0.50 | Model not learning — check loss, LR, or data |
| 0.50 – 0.75 | Rough segmentation — leaves detected but poor boundaries |
| 0.75 – 0.90 | Good segmentation — minor boundary errors |
| 0.90 – 0.95 | High quality — suitable for most applications |
| > 0.95 | Near-perfect — limited by label noise |

### Watershed Quality

| Leaves Detected | Interpretation |
|----------------|---------------|
| Exactly correct | Ideal — each leaf is one instance |
| Over-counted | Over-segmentation — leaves split by noise |
| Under-counted | Under-segmentation — closely overlapping leaves merged |
| Single blob | Watershed threshold too low — increase to 0.5× max |

---

## 8. Common Issues & Troubleshooting

### Issue 1: Loss = NaN

**Symptoms:** Training reports `loss=NaN (skip)` for many batches.

**Causes:**
1. AMP float16 overflow in softmax → CrossEntropy
2. Learning rate too high for RMSprop
3. Very small foreground in a batch (all-background samples)

**Fixes:**
1. Disable AMP (`AMP = False`) — trades speed for stability
2. Reduce learning rate (`LR = 5e-6`)
3. Add numerical epsilon to loss computation

### Issue 2: Model Collapse (Dice → 0 after good epochs)

**Symptoms:** Dice drops from 0.96 to 0.00 around epoch 13.

**Root cause:** Without class weighting, BCE gradient for background overwhelms foreground gradient. Once the model starts predicting "all background," the gradient becomes self-reinforcing.

**Fix:** Enable class weighting in BCE (already applied in `combined_loss` with weight parameter). If still collapsing:
1. Increase `w_fg` clamp upper bound to `5.0`
2. Or switch to pure Dice loss (remove BCE term)

### Issue 3: Watershed OpenCV Error

```
error: (-209:Sizes of input arguments do not match)
```

**Cause:** The overlay image (original size) and colored label map (resized) have different dimensions.

**Fix:** Resize original image to match the label map dimensions (256×256).

### Issue 4: All IoU/Dice = 0

**Symptoms:** Validation metrics show 0.0000 despite loss decreasing.

**Causes:**
1. `best_dice` initialized to 0.0 → epoch 1 not saved as best
2. All predictions are background (model collapsed)
3. Data mismatch between train and validation splits

**Fixes:**
1. Initialize `best_dice = -1.0` so epoch 1 auto-saves
2. Check binary mapping: `mask > 0`
3. Verify seed consistency in `random_split`

---

## 9. Next Steps

### Immediate Improvements

| Priority | Improvement | Expected Impact |
|----------|------------|----------------|
| P1 | Train with `AMP = False` for 50+ epochs | Higher final Dice > 0.98 |
| P1 | Add test-time augmentation (TTA) | Dice +0.01–0.02 |
| P2 | Tune Watershed threshold per image | Better overlap separation |
| P2 | Add boundary loss term | Sharper leaf edges |

### Model Architecture Upgrades

After binary segmentation is stable (>0.95 Dice), consider:

1. **Attention U-Net** — focus on leaf boundaries for sharper masks
2. **Deep Supervision** — auxiliary losses at each decoder level
3. **Multi-scale input** — better handling of different leaf sizes
4. **Boundary-aware loss** — explicit contour supervision

### Deployment Considerations

- **Export:** Convert model to TorchScript or ONNX for inference
- **Batch prediction:** Use `predict.py` with n_classes=2
- **Watershed integration:** Add as post-processing step in prediction pipeline
- **Leaf counting:** Watershed `labels.max()` gives leaf count per image

---

## Appendix: Quick Reference

### File Structure

```
segmentasi-unet/
├── Unet_Binary_Leaf_Segmentation.ipynb   # Main notebook
├── checkpoints/
│   ├── best_binary.pth                   # Best model (Dice-based)
│   └── binary_unet_complete.pth          # Final model + history
├── unet/
│   ├── unet_model.py                     # U-Net architecture
│   ├── unet_parts.py                     # Building blocks
│   └── __init__.py
├── utils/
│   ├── data_loading.py                   # BasicDataset
│   └── dice_score.py                     # Dice loss/metrics
└── data/
    ├── imgs/                             # 347 RGB images
    └── masks/                            # 347 instance masks
```

### Cell Reference (Notebook)

| Cell | Section | Function |
|------|---------|----------|
| 0-5 | Setup | Clone repo, install deps, download data |
| 7 | Dataset | `BinaryLeafDataset` class definition |
| 10 | Loss | `combined_loss()` with class weighting |
| 12 | Metrics | `binary_metrics()` IoU + Dice |
| 14 | Config | Training hyperparameters |
| 16 | Init | Model, optimizer, loader initialization |
| 17 | Training | Main training loop |
| 19 | Plot | Learning curve visualization |
| 21 | Evaluate | Final evaluation on validation set |
| 23 | Visualize | Sample prediction visualization |
| 25 | Watershed | Watershed separation functions |
| 26 | Demo Watershed | Watershed on validation samples |

---

*Documentation for binary leaf segmentation pipeline. Part of the segmentasi-unet project.*
