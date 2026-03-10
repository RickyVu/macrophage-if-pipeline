# Macrophage Immunofluorescence Segmentation Pipeline

An end-to-end computational pipeline for processing multiplex immunofluorescence (mIF) whole-slide images, extracting individual macrophage cell crops, filtering for quality, training a CNN-based quality classifier, performing single-cell segmentation with Cellpose, and conducting downstream phenotypic analysis.

Built for CellDIVE multiplexed imaging data across multiple patient cohorts, targeting four immunosuppressive macrophage subtypes: **PD1+/PD-L1+ x CD68+/CD163+**.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [File Descriptions](#file-descriptions)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Technical Notes](#technical-notes)

---

## Overview

The pipeline takes large multi-gigabyte OME-TIFF whole-slide images with associated CELESTA cell-type assignment CSVs, and produces per-cell segmentation masks, morphological measurements, channel intensity profiles, and dimensionality reduction visualisations suitable for downstream biological analysis.

**Key design choices:**
- All metadata is tracked in a central SQLite database, enabling incremental, resumable processing
- Parallel I/O and batched writes are used throughout to handle datasets with millions of cell images
- Quality control is two-tiered: a fast rule-based filter eliminates obviously poor images, followed by a CNN trained on human-annotated labels to catch subtle artefacts that scalar metrics miss
- Cellpose is used for instance segmentation, with a smart channel composition strategy that adapts to noisy marker channels

---

## Pipeline Architecture

```
OME-TIFF whole-slide images  +  CELESTA cell-type CSVs
              │
              ▼
  [1] ometiff-crop-thread-multibatch-precise.py
      Crop 50x50 cell patches per macrophage subtype
              │
              ▼
  [2] index_images.py
      Index all crops into SQLite (image_index.sqlite)
              │
              ▼
  [3] filter.py
      Rule-based quality filter (SNR, focus, completeness, signal)
              │
         ┌────┴────────────────────┐
         ▼                         ▼
  [4] viewer.py               [5] manual_inspect.py
   (optional: browse)          Human annotation: good / bad
                                         │
                                         ▼
                               [6] train_classifier_cnn.py
                                   Train 6-channel CNN
                                         │
                                         ▼
                               [7] segment.py
                               CNN classify + Cellpose segment
                               → HDF5 stacks + cell_stats in DB
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                   [8] analyze.py   [9] embed.py   [10] pairwise.py
                   Morphological    CNN embeddings  Per-pair binary
                   feature analysis + analysis      classification
```

---

## File Descriptions

### Data Ingestion

| File | Description |
|------|-------------|
| `ometiff-crop-thread-multibatch-precise.py` | Reads large OME-TIFF whole-slide images and CELESTA cell-type assignment CSVs. For each target macrophage subtype, crops a 50x50 pixel patch centred on the cell centroid from the relevant fluorescence channels. Applies background subtraction and CLAHE contrast enhancement. Outputs one OME-TIFF per cell, organised by patient case and macrophage type. Multi-threaded with configurable worker count and batch-write buffering. |
| `index_images.py` | Walks the cropped-cell directory tree and bulk-inserts file paths, case names, and macrophage types into the `images` table in a SQLite database. Optimised for millions of files via `os.walk`, batch commits, and WAL journal mode. Accepts `config.json` or CLI arguments. |
| `config.json` | Central configuration file. Sets `root_dir` (path to indexed cell images), `db_path` (SQLite file), filtering thresholds, and viewer settings. All scripts read from this file; CLI arguments override config values. |

### Quality Filtering

| File | Description |
|------|-------------|
| `filter.py` | Multi-threaded rule-based quality filter. Reads images from the `images` table and computes five per-image metrics: nucleus signal-to-noise ratio (Otsu-based), focus sharpness (Laplacian variance), cell completeness (centroid offset + nucleus area fraction), membrane channel mean intensity, and marker channel mean intensity. Results and pass/fail decisions are written to the `filtered_images` table. Each worker thread owns exclusive directories to eliminate filesystem lock contention. Supports `--resume` to continue interrupted runs. All thresholds are configurable in `config.json`. |
| `viewer.py` | Interactive OME-TIFF viewer built with OpenCV. Provides five rendering modes: nucleus+membrane RGB composite, macrophage-merged overlay, full per-marker IF composite, single-channel coloured grid, and all-channel mosaic. Used as a utility library by the inspection tools. |

### Human Annotation

| File | Description |
|------|-------------|
| `manual_inspect.py` | Interactive annotation tool for labelling images as `good` or `bad` to build a training dataset for the CNN classifier. Displays images one at a time in a stratified queue that balances annotation effort across macrophage subtypes. Labels are written to `inspected_images` immediately on keypress. Supports five viewing modes, forward/backward navigation, and a configurable per-type target label count. Controls: `G` good, `B` bad, `S` skip, `←/→` navigate, `1-5` change view, `Q` quit. |
| `auto_inspect.py` | Automated inspection tool that programmatically generates quality labels to supplement manual annotations. |

### CNN Training

| File | Description |
|------|-------------|
| `train_classifier_cnn.py` | Trains `CellQualityCNN`, a lightweight 6-channel binary classifier (6→32→64→128 conv blocks, AdaptiveAvgPool, 256-dim FC head). Reads good/bad labels from `inspected_images`, performs a stratified 80/20 train/val split, and trains with Focal Loss (γ=2, label smoothing ε=0.05) and a 3:1 bad:good `WeightedRandomSampler` to handle class imbalance. Optimises F₀.₅ (precision-weighted) on the validation set. After training, sweeps classification thresholds to find the operating point satisfying a configurable minimum precision constraint. Saves the best checkpoint to `cnn_model.pt` and a full metrics JSON alongside it. |

### Segmentation

| File | Description |
|------|-------------|
| `segment.py` | Combined CNN classification and Cellpose segmentation pipeline. Loads quality-passed images from the DB, runs the trained `CellQualityCNN` to discard remaining poor-quality images, then passes CNN-good images through Cellpose (cyto3 model) for single-cell instance segmentation. Selects the cell closest to the image centre from multi-cell predictions. Extracts 21 per-cell statistics: 9 morphological (area, perimeter, eccentricity, axis lengths, solidity, extent, circularity, centroid offset) and 12 intensity (mean and std per channel). Writes results to three DB tables (`classified_images`, `segmented_images`, `cell_stats`) and stacks cell images and masks into compressed HDF5 files organised by case and macrophage type. Uses two-phase parallelism: small directories are processed in parallel across workers; large directories split rows across all workers for intra-directory parallelism. |

### Analysis

| File | Description |
|------|-------------|
| `analyze.py` | Comprehensive feature-based analysis of per-cell morphological and intensity statistics stored in `cell_stats`. Applies feature engineering (aspect ratio, membrane/nucleus ratio, marker sum/max, coefficient of variation per channel), imputes missing values, winsorises at 1st/99th percentile, removes outliers with Isolation Forest, and scales with RobustScaler. Produces: PCA (scatter + loadings + scree), t-SNE, UMAP, Random Forest (5-fold CV with MDI and permutation importance), Kruskal-Wallis tests with Benjamini-Hochberg FDR correction, feature boxplots, Pearson correlation heatmap, Ward hierarchical clustering, and a marker positivity heatmap. |
| `embed.py` | CNN embedding-based analysis. Loads cell images from HDF5 files and passes them through a 6-channel-adapted ImageNet-pretrained ResNet-18 (first-layer weights averaged from RGB to 6 channels, frozen inference) to extract 512-dimensional feature vectors. Saves embeddings to `embeddings.npz`. Runs the same analysis suite as `analyze.py` on the high-dimensional embeddings, capturing spatial and textural structure that scalar morphological summaries cannot represent. |
| `pairwise.py` | Pairwise macrophage-type comparison. For each pair of subtypes, subsamples both groups to `min(N_A, N_B)` for a balanced binary classification problem, then runs: PCA scatter, t-SNE, UMAP, binary Random Forest (5-fold accuracy, balanced accuracy, ROC-AUC), ROC curve, and top-discriminating features bar chart. Operates in two modes: `features` (morphological/intensity features) or `embeddings` (CNN embeddings from `embed.py`). Outputs a `master_comparison.csv` and AUC bar chart summarising separability across all pairs. |

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended for `segment.py`, `embed.py`, and `train_classifier_cnn.py`)
- ~16 GB RAM recommended for large datasets

### Installation

```bash
git clone https://github.com/RickyVu/macrophage-if-pipeline.git
cd macrophage-if-pipeline
pip install -r requirements.txt
```

> **Note:** `torch` and `torchvision` are pinned to CUDA 13.0 builds in `requirements.txt`. If your CUDA version differs, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`.

### Configuration

Edit `config.json` before running any script:

```json
{
  "project": {
    "root_dir": "/path/to/your/cropped/cells",
    "db_path": "image_index.sqlite"
  },
  "filtering": {
    "workers": 16,
    "rules": { ... }
  }
}
```

All scripts fall back to `config.json` in the same directory. Any setting can be overridden with CLI flags (e.g., `--db`, `--config`, `--workers`).

---

## Running the Pipeline

### Step 1 - Crop Cells from OME-TIFF

Edit the `DATASETS` list and `OUTPUT_ROOT` near the top of the script, then run:

```bash
python ometiff-crop-thread-multibatch-precise.py
```

**Input:** OME-TIFF whole-slide images + CELESTA assignment CSVs
**Output:** `<OUTPUT_ROOT>/<case>/<macrophage_type>/*.ome.tiff`

---

### Step 2 - Index Images

```bash
python index_images.py /path/to/cropped/cells --db image_index.sqlite
```

**Output:** `image_index.sqlite` - `images` table populated

---

### Step 3 - Rule-Based Quality Filter

```bash
python filter.py --workers 16

# Resume an interrupted run:
python filter.py --workers 16 --resume
```

**Output:** `filtered_images` table in SQLite

---

### Step 4 (Optional) - Browse Images

```bash
python viewer.py
```

Interactive viewer; no database writes.

---

### Step 5 - Manual Annotation

```bash
python manual_inspect.py --target 1000
```

Aim for at least a few hundred labelled examples per macrophage type. Labels are saved in real time; the session can be stopped and resumed at any point.

**Output:** `inspected_images` table in SQLite

---

### Step 6 - Train CNN Classifier

```bash
python train_classifier_cnn.py --epochs 60 --batch-size 32
```

**Output:** `cnn_model.pt`, `cnn_model.json`

---

### Step 7 - Segment Cells

```bash
python segment.py --cnn-model cnn_model.pt --workers 8 --cp-batch 16

# Resume from a previous interrupted run:
python segment.py --cnn-model cnn_model.pt --workers 8 --resume
```

**Output:**
- `stacked_cells/<case>/<type>/cells.h5` - stacked cell images and masks
- `classified_images`, `segmented_images`, `cell_stats` tables in SQLite

---

### Step 8 - Morphological Feature Analysis

```bash
python analyze.py --out-dir analysis_output/
```

**Output:** `analysis_output/` - plots, CSVs, and `summary.txt`

---

### Step 9 - CNN Embedding Analysis

```bash
python embed.py --h5-dir stacked_cells/ --out-dir embedding_output/

# Re-use saved embeddings without re-running the model:
python embed.py --skip-extract --out-dir embedding_output/
```

**Output:** `embedding_output/` - `embeddings.npz`, plots, `summary.txt`

---

### Step 10 - Pairwise Type Comparison

```bash
# Using morphological features
python pairwise.py --mode features --out-dir pairwise_features/

# Using CNN embeddings
python pairwise.py --mode embeddings \
    --embeddings embedding_output/embeddings.npz \
    --out-dir pairwise_embeddings/
```

**Output:** Per-pair subdirectories with scatter plots, ROC curves, RF reports, and a `master_comparison.csv`

---

## Outputs

### SQLite Database (`image_index.sqlite`)

| Table | Contents |
|-------|----------|
| `images` | Master index: file path, case name, macrophage type |
| `filtered_images` | Rule-based quality metrics and pass/fail per image |
| `inspected_images` | Human-annotated good/bad labels for CNN training |
| `classified_images` | CNN quality classification predictions and confidence |
| `segmented_images` | Cellpose output references (HDF5 path + stack index) |
| `cell_stats` | 21-feature per-cell morphology and intensity statistics |

### HDF5 Cell Stacks (`stacked_cells/`)

```
stacked_cells/
  <case_name>/
    <macrophage_type>/
      cells.h5
        /images     (N, 6, H, W)  float32  normalised to [0, 1]
        /masks      (N, H, W)     uint16   binary segmentation mask
        /image_ids  (N,)          int64    FK → images.id
```

### Analysis Outputs (`analysis_output/`, `embedding_output/`)

```
01_pca_scatter.png              PCA PC1 x PC2 with 1σ ellipses per type
02_pca_loadings.png             PCA loading vectors (analyze.py only)
03_pca_scree.png                Explained variance per component
04_tsne.png                     t-SNE 2D embedding
05_umap.png                     UMAP 2D embedding
06_rf_importance.png            Random Forest feature importance
07_feature_boxplots.png         Top-12 Kruskal-Wallis features (analyze.py only)
08_correlation_heatmap.png      Pearson correlation matrix (analyze.py only)
09_hierarchical_clustering.png  Ward dendrogram coloured by macrophage type
10_marker_positivity_heatmap.png  % positive cells per marker per type
kruskal_wallis.csv              H-statistics and BH-corrected q-values
feature_stats_by_type.csv       Per-feature mean ± std by macrophage type
summary.txt                     Run parameters and key numeric metrics
```

---

## Technical Notes

- **Filesystem performance:** All multi-threaded stages assign exclusive directory ownership to each worker, preventing NTFS directory-entry lock contention that causes progressive slowdown on Windows with millions of small files.
- **Class imbalance handling:** The CNN training dataset was heavily imbalanced (~7:1 bad:good). This is addressed with Focal Loss (γ=2), label smoothing (ε=0.05), and a 3:1 bad:good `WeightedRandomSampler` during training.
- **Threshold selection:** After training, a threshold sweep selects the operating point that maximises F₀.₅ subject to a minimum precision constraint (default 0.90), prioritising precision over recall to minimise downstream contamination from low-quality cells.
- **Adaptive channel composition for Cellpose:** Before segmentation, each channel is assessed for noise using an Otsu-based connected-component area criterion. Only clean channels contribute to the cytoplasm composite; images where all marker channels are noisy fall back to nuclei-only segmentation mode.
- **Pairwise vs. global analysis:** Global multi-class analysis must subsample to the size of the rarest class, discarding a large portion of data for well-represented types. Pairwise analysis subsamples each comparison independently to `min(N_A, N_B)`, preserving statistical power for well-populated pairs while remaining fair to smaller classes.
