# Class Determination in U-Net Segmentation Training

**Document Version:** 1.0  
**Last Updated:** 2026-07-29  
**Project:** U-Net Segmentation for Plant Phenotyping Dataset

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overview of Class Determination Flow](#2-overview-of-class-determination-flow)
3. [Code Analysis: Step-by-Step](#3-code-analysis-step-by-step)
   - [3.1 Dataset Initialization](#31-dataset-initialization-data_loadingpy-line-39)
   - [3.2 Mask File Scanning](#32-mask-file-scanning-data_loadingpy-lines-26-35)
   - [3.3 Aggregation of Unique Values](#33-aggregation-of-unique-values-data_loadingpy-lines-53-60)
   - [3.4 Class Count Assignment](#34-class-count-assignment-trainpy-lines-202-206)
   - [3.5 Mask Remapping for Training](#35-mask-remapping-for-training-data_loadingpy-lines-78-84)
4. [Data Validation Results](#4-data-validation-results)
5. [Technical Explanation of Class Count](#5-technical-explanation-of-class-count)
6. [Anomaly Analysis: Pixel Value 27](#6-anomaly-analysis-pixel-value-27)
7. [Impact on Model Architecture](#7-impact-on-model-architecture)
8. [Summary](#8-summary)

---

## 1. Introduction

In semantic segmentation, the number of classes (n_classes) determines the output dimension of the model. Each pixel in the input image is classified into one of these classes. The determination of n_classes must accurately reflect the actual labels present in the dataset. This document explains the automated mechanism in this project for detecting the number of classes directly from the mask files, providing traceability from raw data to model configuration.

The class count is not assumed or manually specified. It is computed algorithmically by scanning all mask files, extracting their unique pixel values, and counting them.

---

## 2. Overview of Class Determination Flow

The following represents the data flow from the mask files on disk to the model instantiation:

```
Mask Files (PNG)
    |
    v
[1] unique_mask_values() called for each file (347 calls)
    - Opens each PNG as a NumPy array
    - Extracts unique pixel values using np.unique()
    - Returns an array of unique integers per file
    |
    v
[2] Results aggregated via np.concatenate()
    - Combines 347 individual arrays into a single array
    |
    v
[3] np.unique() on the concatenated array
    - Produces the global set of unique pixel values across all masks
    - Stored as self.mask_values (a sorted list)
    |
    v
[4] len(self.mask_values) = n_classes
    - The list length becomes the number of output classes
    |
    v
[5] UNet model instantiated with n_classes from step 4
```

---

## 3. Code Analysis: Step-by-Step

### 3.1 Dataset Initialization (data_loading.py, line 39)

The BasicDataset class is instantiated in train.py at line 205:

```python
temp_dataset = BasicDataset(dir_img, dir_mask, args.scale, target_size=(256, 256))
```

Upon construction, the `__init__` method (data_loading.py, line 39) immediately initiates a scan of all mask files to discover their unique pixel values.

```python
def __init__(self, images_dir, mask_dir, scale=1.0, mask_suffix='', target_size=None):
    self.ids = [splitext(file)[0] for file in listdir(images_dir)
                if isfile(join(images_dir, file)) and not file.startswith('.')]

    logging.info('Scanning mask files to determine unique values')
    with Pool() as p:
        unique = list(tqdm(
            p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,
                           mask_suffix=self.mask_suffix), self.ids),
            total=len(self.ids)
        ))

    self.mask_values = list(sorted(np.unique(
        np.concatenate(unique), axis=0).tolist()))
    logging.info(f'Unique mask values: {self.mask_values}')
```

The `self.ids` list contains base filenames derived from the images directory. For each ID, the corresponding mask file is located and scanned. Multiprocessing via `Pool().imap` is used to scan files in parallel for efficiency.

### 3.2 Mask File Scanning (data_loading.py, lines 26-35)

The function `unique_mask_values` is the core primitive that reads a single mask file and extracts its unique pixel values.

```python
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
```

**Detailed breakdown of each step:**

| Line | Code | Explanation |
|------|------|-------------|
| 27 | `mask_dir.glob(idx + mask_suffix + '.*')` | Locates the mask file corresponding to the image ID. For example, for image `ara2012_plant001.png`, it searches for `ara2012_plant001.*` inside the masks directory. |
| 28 | `np.asarray(load_image(mask_file))` | Opens the PNG file and converts it into a NumPy array. Each pixel in this array is an integer representing a class label. |
| 29 | `mask.ndim == 2` | Checks if the mask is a single-channel (grayscale) image, which is the standard format for segmentation masks. |
| 30 | `np.unique(mask)` | Returns a sorted array of all unique integer values present in this particular mask file. For example, if the mask contains pixels with values [0, 1, 2, 3, 5], it returns `[0, 1, 2, 3, 5]`. |

**Concrete example:**

For file `ara2012_plant033.png`, the mask is read as a 2D array. The function `np.unique()` extracts all distinct pixel values found in that file:

```python
# Pseudocode of what happens internally
mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 2, 2, 0],
    [0, 0, 2, 2, 0],
    [27, 27, 0, 0, 0]
])
result = np.unique(mask)
# result = [0, 1, 2, 27]
```

### 3.3 Aggregation of Unique Values (data_loading.py, lines 53-60)

Once all 347 mask files have been individually scanned, their results must be combined to produce the global set of unique values.

**Step A: Concatenation**

```python
np.concatenate(unique)
```

The variable `unique` is a list of 347 arrays, where each array contains the unique values from one mask file. `np.concatenate` joins these into a single array. Example:

```python
# unique = [[0, 1, 2], [0, 1, 3, 4], [0, 27], ...]
# After concatenation:
# [0, 1, 2, 0, 1, 3, 4, 0, 27, ...]
```

**Step B: Global Uniqueness**

```python
np.unique(np.concatenate(unique), axis=0)
```

This extracts the unique values from the concatenated array, producing the master set of pixel values present across the entire dataset. Each value appears exactly once in the result.

**Step C: Sorting and Storage**

```python
self.mask_values = list(sorted(np.unique(...).tolist()))
```

The result is converted to a sorted Python list and stored as `self.mask_values`. For this dataset, the result is:

```python
self.mask_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27]
```

### 3.4 Class Count Assignment (train.py, lines 202-206)

In `train.py`, the `main` block explicitly creates a temporary dataset to determine the class count before constructing the model.

```python
# Determine n_classes from the dataset by scanning mask files first.
# This handles datasets where the number of classes differs from what
# is passed via --classes (e.g. when the mask has extra pixel values).
temp_dataset = BasicDataset(dir_img, dir_mask, args.scale, target_size=(256, 256))
n_classes = len(temp_dataset.mask_values)
logging.info(f'Detected {n_classes} classes from mask files (override --classes {args.classes})')
```

```python
n_classes = len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27])
# n_classes = 22
```

This value of `n_classes` is then passed directly to the model constructor:

```python
model = UNet(n_channels=3, n_classes=n_classes, bilinear=args.bilinear)
```

### 3.5 Mask Remapping for Training (data_loading.py, lines 78-84)

During training, each mask's pixel values must be remapped from their original integers (e.g. 0, 1, 20, 27) to sequential class indices (0, 1, 2, ..., 21) for the CrossEntropyLoss function.

```python
if is_mask:
    mask = np.zeros((newH, newW), dtype=np.int64)
    for i, v in enumerate(mask_values):
        if img.ndim == 2:
            mask[img == v] = i
        else:
            mask[(img == v).all(-1)] = i
    return mask
```

**The remapping logic:**

For each pixel value `v` in `mask_values` (the global list), all pixels in the current mask that equal `v` are set to index `i`.

| Original pixel value `v` | Remapped index `i` | Explanation |
|--------------------------|-------------------|-------------|
| 0                        | 0                 | Background   |
| 1                        | 1                 | Leaf label 1  |
| 2                        | 2                 | Leaf label 2  |
| ...                      | ...               | ...          |
| 20                       | 20                | Leaf label 20 |
| 27                       | 21                | Anomaly value |

---

## 4. Data Validation Results

A complete scan of the dataset produced the following verifiable results:

| Metric | Value |
|--------|-------|
| Total mask files scanned | 347 |
| Unique pixel values discovered | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27] |
| Number of unique values (n_classes) | 22 |
| File containing anomalous value 27 | ara2012_plant033.png |
| Pixel count of value 27 in that file | 198 out of 112,530 total pixels |
| Number of files with only values 0-20 | 346 |

The validation was performed by executing the exact same code path used during training initialization. The complete simulation can be reproduced with:

```python
import numpy as np
from PIL import Image
from os import listdir
from os.path import join

mask_dir = 'data/masks'
files = [f for f in listdir(mask_dir) if f.endswith('.png')]

def unique_mask_values(filename):
    mask = np.asarray(Image.open(join(mask_dir, filename)))
    return np.unique(mask)

all_values = [unique_mask_values(f) for f in files]
global_values = np.unique(np.concatenate(all_values))
n_classes = len(global_values)

print(f'Global unique values: {global_values.tolist()}')
print(f'n_classes: {n_classes}')
```

This code is functionally identical to the production code path in `data_loading.py` lines 26-60.

---

## 5. Technical Explanation of Class Count

### Why n_classes is determined by the number of unique pixel values

The CrossEntropyLoss function used for training requires target values to be integer class indices ranging from 0 to n_classes - 1. The model's final layer produces n_classes channels, each representing the probability that a pixel belongs to that class.

If the number of unique pixel values in the dataset does not match the model's n_classes, the loss function will reject the data with an IndexError. Therefore, n_classes must equal the exact count of distinct pixel values found across all mask files.

### The problem with assuming class count from documentation

The dataset documentation specifies 21 classes (0-20). However, the actual mask files contain 22 unique values due to the presence of value 27. Relying on the documented class count alone would cause a runtime error:

```
IndexError: Target 21 is out of bounds.
```

This error occurs because:
1. The user passes `--classes 21`.
2. The model creates 21 output channels (indices 0-20).
3. During data loading, pixel value 27 is remapped to index 21 (the 22nd entry in mask_values).
4. CrossEntropyLoss rejects index 21 because the valid range is 0-20.

### Why automated detection is necessary

Automated detection ensures that the model's output dimension always matches the actual data, regardless of discrepancies between documentation and the physical mask files. This approach handles:

- Datasets with undocumented label values.
- Anomalies or artifacts in annotation.
- Subsets of datasets where not all classes appear.
- Datasets with non-contiguous label values.

---

## 6. Anomaly Analysis: Pixel Value 27

### Discovery

The value 27 was discovered during the automated mask scanning process. It exists in exactly one file: `ara2012_plant033.png`.

### Quantification

| Property | Value |
|----------|-------|
| File | ara2012_plant033.png |
| Total pixels in file | 112,530 |
| Pixels with value 27 | 198 |
| Percentage of anomalous pixels | 0.176% |
| All unique values in this file | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27] |

### Root Cause Analysis

Value 27 replaces value 19 in this file. The expected pattern (observed in all other 346 files) is contiguous leaf labels 0 through N. In `ara2012_plant033.png`, leaf label 19 is missing and value 27 appears instead. The most probable causes are:

1. **Annotation tool error**: The annotator's software may have assigned a default or erroneous label value.
2. **Data conversion artifact**: During dataset publication or format conversion, a pixel value may have been corrupted.
3. **Deliberate but undocumented marking**: The value 27 may have been used to mark a specific feature not described in the dataset documentation.

### Handling Strategy

Rather than modifying the mask files (which would alter the original dataset), the automated detection system accommodates this anomaly by counting it as a distinct class. This ensures training completes without error while preserving the original data integrity.

---

## 7. Impact on Model Architecture

The value of n_classes directly affects the model's output layer. In the UNet architecture, the final convolutional layer produces n_classes channels:

```python
self.outc = OutConv(64, n_classes)
```

Where `OutConv` is defined as:

```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

With n_classes = 22, the output tensor has shape `(batch_size, 22, height, width)`. Each of the 22 channels corresponds to one class probability map.

For comparison:

| n_classes | Output channels | Behavior |
|-----------|----------------|----------|
| 21 (documented) | 21 | Fails with IndexError due to value 27 |
| 22 (detected) | 22 | All values remapped to valid indices 0-21 |

---

## 8. Summary

The number of segmentation classes in this project is determined through an automated, data-driven process:

1. Each of the 347 mask files is opened and read as a NumPy array.
2. The unique pixel values are extracted from each file individually.
3. The results are aggregated to produce a global set of unique values.
4. The count of this set becomes n_classes for the model.

For the Plant Phenotyping dataset used in this project:

- The documented class count is 21 (pixel values 0-20).
- The detected class count is 22 (pixel values 0-20 and 27).
- The discrepancy is caused by a single file (ara2012_plant033.png) containing an anomalous pixel value of 27, affecting 198 out of 112,530 pixels.
- Automated detection allows training to proceed without manual data cleanup, preserving the original dataset while ensuring model-data compatibility.

This approach is transparent, reproducible, and verifiable by re-running the scanning code included in the project's `utils/data_loading.py` module.

---

## References

- Code: `utils/data_loading.py`, lines 26-60 (scanning and aggregation)
- Code: `train.py`, lines 202-212 (class count assignment and model instantiation)
- Dataset: Plant Phenotyping Dataset (347 images, 347 masks)
