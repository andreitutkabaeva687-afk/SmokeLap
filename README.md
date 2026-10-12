# SmokeLap: Smoke-Aware Surgical Scene Segmentation

This repository provides the inference and evaluation resources for SmokeLap, a semantic segmentation method designed for surgical scenes affected by smoke.

SmokeLap integrates smoke estimation, temporal memory retrieval and smoke-guided prediction fusion. The repository provides the checkpoint-compatible model definition, SmokeLap dataset loader, inference pipeline, prediction visualization tools and global-confusion-matrix evaluation used for the reported test result.

## Reported result

The evaluation protocol is fixed as follows:

> Accumulate all valid pixels from all 3,386 test images into one 9-class confusion matrix, calculate per-class Dice and IoU, and then macro-average across all 9 classes.

| Split / smoke level | N | Dice | mIoU | PixelAcc |
|---|---:|---:|---:|---:|
| Overall | 3386 | **0.4323** | **0.3062** | 0.5395 |
| L0 | 1716 | 0.4819 | 0.3470 | 0.5736 |
| L1 | 1161 | 0.3718 | 0.2656 | 0.5246 |
| L2 | 390 | 0.3563 | 0.2451 | 0.4723 |
| L3 | 119 | 0.2129 | 0.1415 | 0.4130 |

The corresponding best validation Dice is `0.4473`. The evaluation script reproduces the result files and confusion matrices locally.

## Repository structure

```text
.
├── configs/                 Inference/evaluation configuration
├── datasets/                SmokeLap dataset loader
├── models/                  Checkpoint-compatible inference model
├── scripts/                 Evaluation, prediction and result-export tools
├── utils/                   Configuration and checkpoint utilities
├── checkpoints/README.md    Checkpoint location and checksum
├── requirements.txt
└── run_test_4323.ps1
```

## Installation

```bash
git clone https://github.com/andreitutkabaeva687-afk/SmokeLap.git
cd SmokeLap
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The dataset is not distributed with this repository. The supplied configuration expects:

```text
D:/data/
├── images/
├── masks/
└── smoke/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Edit the paths under `data:` in `configs/smokelap_m20.yaml` if your local dataset is stored elsewhere.

## Pretrained checkpoint

Download the GitHub Release asset and place it at:

```text
checkpoints/smokelap_m20/smokelap_m20_best.pth
```

Expected SHA-256:

```text
85A93829E074631AA8AF0D8D37B22A70F4F6CD32D67E6A06FEDD44A061A54204
```

## Reproduce the test result

```powershell
.\run_test_4323.ps1
```

Equivalent command:

```powershell
python scripts\evaluate_global_confusion.py `
  --config configs\smokelap_m20.yaml `
  --ckpt checkpoints\smokelap_m20\smokelap_m20_best.pth `
  --split test `
  --output_dir outputs\global_confusion\smokelap_m20_best\test `
  --batch_size 1 `
  --num_workers 0 `
  --device cuda
```

Do not enable `--legacy_smoke_fusion` for the released checkpoint.

## Export prediction masks and overlays

```powershell
python scripts\predict_masks_and_overlays.py `
  --config configs\smokelap_m20.yaml `
  --ckpt checkpoints\smokelap_m20\smokelap_m20_best.pth `
  --split test `
  --output_dir outputs\predictions\smokelap_m20_best\test `
  --batch_size 1 `
  --num_workers 0 `
  --device cuda `
  --alpha 0.50 `
  --overlay_format png
```

## Reproducibility details

- Input resolution: `512 × 512`
- Semantic classes: `9`
- Memory-bank size: `20`
- Retrieval top-k: `3`
- Test images: `3,386`
- Metric: 9-class macro-average from one global confusion matrix


