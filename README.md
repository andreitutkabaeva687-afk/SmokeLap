# SmokeLap: Smoke-Aware Surgical Scene Segmentation

This repository is the **inference and evaluation release** of SmokeLap for semantic segmentation under surgical smoke.

SmokeLap integrates smoke estimation, temporal memory retrieval and smoke-guided prediction fusion. This public package provides the trained-model definition, dataset loader, inference pipeline, prediction visualization and the exact global-confusion-matrix evaluation used for the reported test result. Training, optimization and ablation source code are intentionally not included in this release.

## Reported result

The evaluation protocol is fixed as follows:

> accumulate all valid pixels from all 3,386 test images into one 9-class confusion matrix → calculate per-class Dice/IoU → macro-average all 9 classes.

| Split / smoke level | N | Dice | mIoU | PixelAcc |
|---|---:|---:|---:|---:|
| Overall | 3386 | **0.4323** | **0.3062** | 0.5395 |
| L0 | 1716 | 0.4819 | 0.3470 | 0.5736 |
| L1 | 1161 | 0.3718 | 0.2656 | 0.5246 |
| L2 | 390 | 0.3563 | 0.2451 | 0.4723 |
| L3 | 119 | 0.2129 | 0.1415 | 0.4130 |

The corresponding best validation Dice is `0.4473`. The evaluation script reproduces the result files and confusion matrices locally.

## Public-release scope

Included:

- minimal model code required to load the released checkpoint;
- SmokeLap dataset loader;
- full-test global confusion-matrix evaluator;
- mask and overlay prediction script;
- inference configuration;

Not included:

- training and optimization pipeline;
- loss-function implementation;
- ablation-training code;
- experimental V3/V4/V5 branches;
- private experiment queues and logs.

## Repository structure

```text
.
├── configs/                 Inference/evaluation configuration
├── datasets/                SmokeLap dataset loader
├── models/                  Minimal checkpoint-compatible inference model
├── scripts/                 Evaluation, prediction and result-export tools
├── utils/                   Configuration and checkpoint utilities
├── checkpoints/README.md    Checkpoint location and checksum
├── requirements.txt
└── run_test_4323.ps1
```

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
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

## License

No license is granted until the repository owner adds a license file. Add the license required by your institution and dataset agreement before making this repository public.
