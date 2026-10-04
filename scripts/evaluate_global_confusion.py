# -*- coding: utf-8 -*-


import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.legacy_smokelap_runtime import install_legacy_smoke_fusion  # noqa: E402


CLASS_NAMES = [
    "background", "liver", "gallbladder", "other_tissue", "duct",
    "instrument_metal", "gauze", "hook", "clip",
]
LEVELS = ("L0", "L1", "L2", "L3")
ENDOVIT = {"mDice": 0.3610, "mIoU": 0.2875, "Drop": 0.0967, "AUSC": 0.3326}


def parse_args():
    parser = argparse.ArgumentParser(description="Global confusion-matrix evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--legacy_smoke_fusion",
        action="store_true",
        help="Restore the smoke-guided fusion used by earlier SmokeLap checkpoints",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def update_cm(cm, pred, target, num_classes, ignore_index):
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = (
        (target != ignore_index) & (target >= 0) & (target < num_classes)
        & (pred >= 0) & (pred < num_classes)
    )
    indices = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    cm += np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def calculate_metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    iou_den = gt + pred - tp
    dice_den = gt + pred
    iou = np.divide(tp, iou_den, out=np.zeros_like(tp), where=iou_den != 0)
    dice = np.divide(2.0 * tp, dice_den, out=np.zeros_like(tp), where=dice_den != 0)
    total = float(cm.sum())
    return {
        "dice": dice,
        "iou": iou,
        "pixel_acc": float(tp.sum() / total) if total else 0.0,
        "mean_dice_all": float(np.mean(dice)),
        "mean_iou_all": float(np.mean(iou)),
        "mean_dice_fg": float(np.mean(dice[1:])),
        "mean_iou_fg": float(np.mean(iou[1:])),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix(path, cm):
    fields = ["gt_pred"] + CLASS_NAMES
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        row = {"gt_pred": name}
        row.update({CLASS_NAMES[j]: int(cm[i, j]) for j in range(len(CLASS_NAMES))})
        rows.append(row)
    write_csv(path, rows, fields)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model_type:
        cfg["model_type"] = args.model_type
    if args.image_size is not None:
        cfg["data"]["image_size"] = int(args.image_size)

    model_type = cfg.get("model_type", "smokelap")
    is_smokelap = model_type == "smokelap"
    device = resolve_device(args.device)
    checkpoint = Path(args.ckpt).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else
        PROJECT_ROOT / "outputs" / "global_confusion" / checkpoint.stem / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg, args.split, PROJECT_ROOT)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint, model, device, strict=True)
    if args.legacy_smoke_fusion:
        if not is_smokelap:
            raise ValueError("--legacy_smoke_fusion is valid only for SmokeLap models")
        install_legacy_smoke_fusion(model)
    model.eval()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    num_classes = int(cfg["data"]["num_classes"])
    ignore_index = int(cfg["data"].get("ignore_index", 255))
    if num_classes != 9:
        raise ValueError(f"This protocol requires 9 classes, got {num_classes}")

    overall_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    level_cms = {x: np.zeros_like(overall_cm) for x in LEVELS}
    level_counts = {x: 0 for x in LEVELS}

    print("=" * 80)
    print("SmokeLap GLOBAL CONFUSION Evaluation")
    print("All pixels -> one CM -> per-class Dice/IoU -> 9-class mean")
    print("=" * 80)
    print("Device     :", device)
    print("Checkpoint :", checkpoint)
    print("Samples    :", len(dataset))
    print("Image size :", cfg["data"]["image_size"])
    print("Batch size :", args.batch_size)
    print("Legacy fusion:", bool(args.legacy_smoke_fusion))

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Global confusion evaluate"):
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            masks = batch["mask"].to(device, non_blocking=device.type == "cuda")
            if is_smokelap:
                logits = model(
                    images, smoke_scores_gt=None, video_ids=batch["video_id"]
                )["logits"]
            elif model_type == "unetpp":
                logits = model(images[:, -1])
            elif model_type == "convlstm":
                logits = model(images)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            for i in range(len(preds)):
                level = str(batch["smoke_level"][i])
                update_cm(overall_cm, preds[i], targets[i], num_classes, ignore_index)
                update_cm(level_cms[level], preds[i], targets[i], num_classes, ignore_index)
                level_counts[level] += 1

    overall = calculate_metrics(overall_cm)
    level_metrics = {x: calculate_metrics(level_cms[x]) for x in LEVELS}
    d = [level_metrics[x]["mean_dice_all"] for x in LEVELS]
    drop = float(d[0] - d[3])
    ausc = float((d[0] + 2 * d[1] + 2 * d[2] + d[3]) / 6.0)

    print("\n" + "=" * 80)
    print("GLOBAL CONFUSION-MATRIX RESULTS")
    print("=" * 80)
    print(f"Pixel Accuracy : {overall['pixel_acc']:.4f}")
    print(f"mDice (9 cls)  : {overall['mean_dice_all']:.4f}")
    print(f"mIoU  (9 cls)  : {overall['mean_iou_all']:.4f}")
    print(f"mDice (FG)     : {overall['mean_dice_fg']:.4f}")
    print(f"mIoU  (FG)     : {overall['mean_iou_fg']:.4f}")

    summary_rows = [{
        "Level": "Overall", "N": len(dataset),
        "mDice_9cls": overall["mean_dice_all"],
        "mIoU_9cls": overall["mean_iou_all"], "PixelAcc": overall["pixel_acc"],
    }]
    for level in LEVELS:
        metric = level_metrics[level]
        summary_rows.append({
            "Level": level, "N": level_counts[level],
            "mDice_9cls": metric["mean_dice_all"],
            "mIoU_9cls": metric["mean_iou_all"], "PixelAcc": metric["pixel_acc"],
        })
        print(
            f"{level}: N={level_counts[level]} | mDice={metric['mean_dice_all']:.4f} | "
            f"mIoU={metric['mean_iou_all']:.4f} | PixelAcc={metric['pixel_acc']:.4f}"
        )
    print(f"L0->L3 Drop    : {drop:.4f}")
    print(f"AUSC           : {ausc:.4f}")

    class_rows = []
    print("\nPer-class global results:")
    for c, name in enumerate(CLASS_NAMES):
        row = {
            "class_id": c, "class_name": name,
            "Dice": float(overall["dice"][c]), "IoU": float(overall["iou"][c]),
            "GT_pixels": int(overall_cm[c, :].sum()),
            "Pred_pixels": int(overall_cm[:, c].sum()), "TP": int(overall_cm[c, c]),
        }
        class_rows.append(row)
        print(f"{name:<20} Dice={row['Dice']:.4f} | IoU={row['IoU']:.4f}")

    benchmark_rows = [
        {"Metric": "mDice", "Current": overall["mean_dice_all"], "EndoViT": ENDOVIT["mDice"], "Pass": overall["mean_dice_all"] > ENDOVIT["mDice"]},
        {"Metric": "mIoU", "Current": overall["mean_iou_all"], "EndoViT": ENDOVIT["mIoU"], "Pass": overall["mean_iou_all"] > ENDOVIT["mIoU"]},
        {"Metric": "L0->L3 Drop", "Current": drop, "EndoViT": ENDOVIT["Drop"], "Pass": drop < ENDOVIT["Drop"]},
        {"Metric": "AUSC", "Current": ausc, "EndoViT": ENDOVIT["AUSC"], "Pass": ausc > ENDOVIT["AUSC"]},
    ]
    write_csv(output_dir / "global_summary.csv", summary_rows, ["Level", "N", "mDice_9cls", "mIoU_9cls", "PixelAcc"])
    write_csv(output_dir / "global_per_class.csv", class_rows, ["class_id", "class_name", "Dice", "IoU", "GT_pixels", "Pred_pixels", "TP"])
    write_csv(output_dir / "endovit_comparison.csv", benchmark_rows, ["Metric", "Current", "EndoViT", "Pass"])
    save_matrix(output_dir / "global_confusion_matrix.csv", overall_cm)
    for level in LEVELS:
        save_matrix(output_dir / f"{level}_confusion_matrix.csv", level_cms[level])

    metadata = {
        "protocol": "all pixels -> one confusion matrix -> per-class metrics -> 9-class mean",
        "checkpoint": str(checkpoint), "split": args.split, "samples": len(dataset),
        "image_size": int(cfg["data"]["image_size"]), "batch_size": args.batch_size,
        "drop": drop, "ausc": ausc,
        "legacy_smoke_fusion": bool(args.legacy_smoke_fusion),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
