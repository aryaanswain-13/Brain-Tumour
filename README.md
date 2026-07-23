# 3D Brain Tumor Segmentation and Gated-Fusion Multi-Task Classification

This repository contains a clean, modular refactoring of the medical imaging pipeline developed for the **UCSF-PDGM-v5 dataset**. The system utilizes a fine-tuned **3D SwinUNETR** model as a feature extractor to feed a multi-task classifier that predicts molecular and histopathological markers of brain tumors (IDH Mutation Status, WHO CNS Grade, and MGMT Promoter Methylation).

---

## 🚀 Pipeline Overview

```
                          +-----------------------------------+
                          |      Raw 3D Modal NIfTI Scans      |
                          |  (T1, T1c, T2, FLAIR) + Segmentation|
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |   MONAI Normalize & Preprocessing  |
                          +-----------------+-----------------+
                                            |
                                            v
                          +-----------------------------------+
                          |     3D SwinUNETR Segmentation     |
                          +--------+-----------------+--------+
                                   |                 |
                                   |                 v
                                   |      +---------------------+
                                   |      | Test-Time Augment   |
                                   |      | (TTA) Predictions   |
                                   |      +----------+----------+
                                   v                 |
                         +-------------------+       v
                         | Stage 4 Bottleneck| +-----------+
                         | Encoder Features  | |Confidence |
                         |   (GAP) [768]     | | Map [1]   |
                         +---------+---------+ +-----+-----+
                                   |                 |
                                   |    +------------+
                                   v    v
                    +-----------------------------+
                    |   Confidence-Gated Fusion   | <----+ Clamped Morphology
                    |  (Ablation Modes A/B/C/D)   |      |  Features [8]
                    +--------------+--------------+      |
                                   |                     |
                                   v                     |
                         +-------------------+           |
                         |   Joint Fused     |           |
                         | Representation    |           |
                         +---------+---------+           |
                                   |                     |
                                   v                     |
                         +-------------------+           |
                         | Multi-Task Head   |-----------+
                         | (IDH/Grade/MGMT)  |
                         +-------------------+
```

1. **3D Modal Image Pipeline**: Re-orient scans to RAS coordinate space, interpolate voxels to isotropic spacing ($1.0 \text{ mm}^3$), normalize intensities, crop background regions, and pad/crop to $128 \times 128 \times 128$ dimensions.
2. **SwinUNETR Segmentation**: Segments tumors into three tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
3. **Bottleneck Feature Extraction**: Extracts a $768$-dimensional latent vector from the Stage 4 encoder bottleneck of SwinUNETR via global average pooling.
4. **Morphological Characterization**: Computes 3D shape descriptors: volume sizes, boundary surface areas, compactness, sphericity, and edema-to-core ratios.
5. **Confidence Gated Fusion**: A gating module modulates joint features based on prediction confidence scoring maps and spatial morphological parameters.
6. **Task Classification Head**: Maps the fused representations through feed-forward heads predicting IDH mutation, WHO grade, and MGMT promoter status.

---

## 📁 Repository Structure

```
├── README.md                 # Project details & setup documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignoring dataset directories, caches, and weights
├── configs/
│   └── config.py             # Hyper-parameters and path variables
├── src/
│   ├── data/
│   │   ├── dataset.py        # Custom datasets (SegDataset, CachedDataset) & splits
│   │   └── transforms.py     # 3D MONAI preprocess / augmentation pipelines
│   ├── models/
│   │   ├── backbone.py       # SwinUNETR loaders, feature extraction, and TTA functions
│   │   ├── fusion.py         # ConfidenceGatedFusion module supporting gating ablation
│   │   └── classifier.py     # Multi-task BrainTumorClassifier head
│   ├── training/
│   │   ├── losses.py         # Dice Loss & Class-weighted classification losses
│   │   ├── finetune.py       # SwinUNETR segmentation fine-tuning script
│   │   └── train_clf.py      # Classifier head training script with early stopping
│   ├── evaluation/
│   │   ├── metrics.py        # Segmentation (Dice, IoU, HD95) & classification metrics
│   │   └── evaluator.py      # Evaluators, Gating Ablations, and 5-Fold Cross Validation
│   └── utils/
│       └── helpers.py        # Seed controls & morphology calculation utilities
├── scripts/
│   ├── run_finetune.py       # CLI segmentation fine-tuning script
│   ├── run_precompute.py     # CLI feature cache generation script
│   └── run_pipeline.py       # CLI classification training, ablation & evaluation pipeline
└── notebooks/
    └── brain_tumor_pipeline.ipynb # Visual walkthrough notebook replicating the pipeline
```

---

## ⚙️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd brain-tumor-gated-fusion
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Requirements**:
   Place the UCSF-PDGM dataset folder and metadata CSV inside the project directories (or configure paths inside `configs/config.py`):
   * Default metadata file: `configs/config.py` targets `data/UCSF-PDGM-metadata_v5.csv`
   * Default NIfTI directory: `configs/config.py` targets `data/UCSF-PDGM-v5/`

---

## 🏃 Running the Code

### 1. Fine-tuning the Segmentation Backbone
To run fine-tuning on the SwinUNETR late encoder/decoder stages:
```bash
python scripts/run_finetune.py --epochs 100 --lr 1e-4
```

### 2. Precomputing Feature Cache (Recommended)
Precomputing latent representations and shape metrics reduces classifier training time to seconds by avoiding redundant 3D network execution.
```bash
python scripts/run_precompute.py --tta_runs 5
```
This builds:
* `feature_cache_3d_ucsf_finetuned/` (full cache including 3D maps)
* `feature_cache_slim_ucsf/` (lightweight feature descriptors ~1.5 MB total for 487 patients)

### 3. Training & Evaluating Classifier
To train classifier variants across multiple seeds, evaluate gating ablations, run 5-Fold Cross Validation, and execute McNemar's tests:
```bash
python scripts/run_pipeline.py --epochs 60 --seeds "0,1,2,3,4"
```

---

## 📊 Summary of Baseline Results

### 3D Segmentation Metrics (Test Set)
* **Whole Tumor (WT)**: Dice = $0.859 \pm 0.183$ | IoU = $0.783 \pm 0.196$ | HD95 = $9.0 \pm 13.5 \text{ mm}$
* **Tumor Core (TC)**: Dice = $0.758 \pm 0.348$ | IoU = $0.707 \pm 0.345$ | HD95 = $6.3 \pm 10.6 \text{ mm}$
* **Enhancing Tumor (ET)**: Dice = $0.727 \pm 0.335$ | IoU = $0.654 \pm 0.317$ | HD95 = $5.7 \pm 10.2 \text{ mm}$

### Classification Performance (Aggregate Mean Across 5 Seeds)
* **IDH Status Mutation**: AUC = **$0.917 \pm 0.006$** (Accuracy = $0.889$)
* **WHO CNS Grade**: AUC = **$0.976 \pm 0.001$** (Accuracy = $0.922$)
* **MGMT Promoter Methylation**: AUC = **$0.427 \pm 0.006$** (Accuracy = $0.532$)
