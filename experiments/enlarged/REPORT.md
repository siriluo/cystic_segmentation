# Enlarged-dataset training report

**Date:** 1 September 2026  
**Checkpoint:** `experiments/enlarged/deeplabv3_best.pth`  
**Job:** Slurm 3063984 on NVIDIA GH200 (DeltaAI)

## 1. Data cleanup

The upload `AI project case pictures.zip` is 23 GB, almost entirely videos and office files. Only images plus VIA annotations were unpacked (`data/raw_import/`, 967 files, ~195 MB).

Each leaf folder was paired to a **raw laparoscopic frame** (not an annotation overlay) and a VIA JSON/TXT file. Overlay names (`annotation`, `labels`, `スライド`) were dropped. Duplicate frames were removed by SHA-1 of the image bytes. Patient IDs were taken from the deepest `CaseNN` / `CaseXNN` path component so `Case_08-22/Case22` is Case 22, not Case 8, and Juntendo folders `X12_…` become `CaseX12`.

| Filter | Count |
| --- | --- |
| Valid image–annotation pairs | 615 |
| Exact duplicate frames | 207 |
| Annotation with no raw image in the folder | 350 |
| Unreadable JSON | 2 |
| **Unique frames kept** | **408** |
| **Patients** | **58** |

The 350 “missing image” rows are mostly (a) VIA project dumps that list many filenames in one JSON while the folder only has one frame, and (b) Cases 8–16, which only contain overlay PNGs (`4A labels.png`, etc.). Those overlays were **not** used for training.

Compared with the previous local set (169 frames, 3 patients), this is about **2.4× frames** and **19× patients**. Gallbladder is now labeled (the old loader missed the `Gallbradder` typo). Class presence in the 408 frames:

| Class | Frames |
| --- | --- |
| Liver edge | 407 |
| Gallbladder | 393 |
| Cystic duct | 390 |
| Common bile duct | 356 |
| Instrument | 348 |
| Cystic artery | 201 |
| Rouvière's sulcus | 79 |

Processed layout: `data/processed/<CaseID>/<sample>/image.jpg` + `annotation.json`. Manifest: `data/processed/manifest.csv`.

## 2. Training

- **Model:** DeepLabV3-ResNet50, 8 classes (background + 7 landmarks), ImageNet/COCO init  
- **Split:** patient-level, seed 42 — **357 train / 51 val**, 46 vs 11 cases (val: Case19, 21, 26, 29, 30, 33, 45, 48, X09, X10, X13)  
- **Input:** 512×512, flip/rotate/color jitter on train  
- **Loss:** inverse-frequency weighted cross-entropy (helps cystic artery and Rouvière's sulcus)  
- **Optim:** Adam 1e-4, weight decay 1e-4, ReduceLROnPlateau  
- **Early stopping:** best val loss at **epoch 7** (0.884); stopped at epoch 17  

Train loss kept falling after epoch 7 while val loss rose — the saved weights are the epoch-7 checkpoint, not the last epoch.

## 3. Validation metrics

Held-out patients only (51 frames). Per-image mean IoU / Dice:

| Class | IoU | Dice |
| --- | ---: | ---: |
| Background | 0.519 | 0.661 |
| Gallbladder | 0.400 | 0.526 |
| Common bile duct | 0.229 | 0.338 |
| Cystic duct | 0.106 | 0.171 |
| Instrument | 0.523 | 0.654 |
| Liver edge | 0.700 | 0.819 |
| Cystic artery | 0.100 | 0.175 |
| Rouvière's sulcus | 0.022 | 0.038 |
| **Mean (all)** | **0.325** | **0.423** |
| **Mean (foreground)** | **0.297** | **0.389** |

Large, frequent classes (liver edge, instrument, gallbladder) are learned first. Thin, rare ducts and Rouvière's sulcus remain weak — they occupy little of the image and appear in few val cases.

Qualitative overlays (`val_result_*.png`): instruments and liver/gallbladder regions are in the right place; predicted masks fill more of the frame than the sparse expert polygons; cystic duct/artery are often thickened or merged.

## 4. Takeaways

1. The zip is now a single training tree of **408 unique frames / 58 patients**, with consistent labels and no overlay leak.  
2. Retraining on this set produces a usable coarse map of liver, gallbladder, and instruments, but **not** a reliable cystic duct / cystic artery / CBD localizer yet.  
3. Next levers: more frames of Calot’s triangle (especially artery and Rouvière), a boundary-aware loss (Dice + CE), and a lighter real-time backbone if this is meant for the laparoscopic display.

## Artifacts

- Weights: `experiments/enlarged/deeplabv3_best.pth`  
- Metrics: `experiments/enlarged/metrics.json`  
- Split: `experiments/enlarged/split.json`  
- Curves: `experiments/enlarged/loss_curve.png`, `lr_curve.png`  
- Overlays: `experiments/enlarged/val_result_0_*.png`
