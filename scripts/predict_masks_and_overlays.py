# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset  # noqa: E402
from models import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_config  # noqa: E402


CLASS_NAMES = (
    "background",
    "liver",
    "gallbladder",
    "other_tissue",
    "duct",
    "instrument_metal",
    "gauze",
    "hook",
    "clip",
)

PALETTE = np.asarray(
    [
        (0, 0, 0),
        (220, 80, 80),
        (80, 180, 80),
        (230, 180, 80),
        (80, 140, 220),
        (170, 100, 210),
        (80, 200, 200),
        (240, 130, 50),
        (230, 220, 70),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predicted masks and overlays with the SmokeLap palette."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument(
        "--overlay_format",
        default="jpg",
        choices=("jpg", "png"),
        help="JPG is the space-safe default; masks are always lossless PNG.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--blend_background",
        action="store_true",
        help="Blend black background too; by default class 0 leaves the input unchanged.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def safe_component(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text or "unknown"


def source_lookup(dataset) -> dict[tuple[str, int, str], Path]:
    lookup = {}
    for row in dataset.rows:
        name = Path(row["image"]).name
        key = (str(row["video_id"]), int(row["frame_id"]), name)
        lookup[key] = dataset._ip(row["image"])
    return lookup


def resize_ids(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(image.resize(size, resample=Image.Resampling.NEAREST), dtype=np.uint8)


def save_palette(output_dir: Path) -> None:
    with (output_dir / "palette.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "R", "G", "B"])
        for class_id, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
            writer.writerow([class_id, name, *[int(value) for value in color]])

    swatch_height, swatch_width = 52, 360
    legend = Image.new("RGB", (swatch_width, swatch_height * len(CLASS_NAMES)), "white")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(legend)
        for class_id, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
            top = class_id * swatch_height
            draw.rectangle((0, top, 80, top + swatch_height), fill=tuple(int(x) for x in color))
            draw.text((92, top + 17), f"{class_id}  {name}", fill=(0, 0, 0))
        legend.save(output_dir / "palette.png")
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100")

    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = Path(args.ckpt).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    dataset = build_dataset(config, args.split, PROJECT_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sources = source_lookup(dataset)

    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, device, strict=True)
    model.eval()
    if hasattr(model, "reset_memory"):
        model.reset_memory()

    output_dir = Path(args.output_dir).resolve()
    mask_id_dir = output_dir / "mask_id"
    mask_color_dir = output_dir / "mask_color"
    overlay_dir = output_dir / "overlay"
    for directory in (mask_id_dir, mask_color_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_palette(output_dir)

    model_type = config.get("model_type", "smokelap")
    manifest_rows = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predict masks and overlays"):
            images = batch["images"].to(device, non_blocking=device.type == "cuda")
            if model_type == "smokelap":
                logits = model(
                    images,
                    smoke_scores_gt=None,
                    video_ids=batch["video_id"],
                )["logits"]
            elif model_type == "unetpp":
                logits = model(images[:, -1])
            elif model_type == "convlstm":
                logits = model(images)
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            for item_index, prediction in enumerate(predictions):
                video_id = str(batch["video_id"][item_index])
                frame_id = int(batch["frame_id"][item_index])
                name = str(batch["name"][item_index])
                key = (video_id, frame_id, name)
                source_path = sources[key]

                with Image.open(source_path) as source_image:
                    source_rgb = source_image.convert("RGB")
                    original = np.asarray(source_rgb, dtype=np.uint8)
                    original_size = source_rgb.size

                prediction_original = resize_ids(prediction, original_size)
                color_mask = PALETTE[prediction_original]
                blended = original.copy()
                region = (
                    np.ones(prediction_original.shape, dtype=bool)
                    if args.blend_background
                    else prediction_original != 0
                )
                if np.any(region):
                    blended_pixels = (
                        (1.0 - args.alpha) * original[region].astype(np.float32)
                        + args.alpha * color_mask[region].astype(np.float32)
                    )
                    blended[region] = np.clip(blended_pixels, 0, 255).astype(np.uint8)

                video_dir = safe_component(video_id)
                output_name = Path(name).with_suffix(".png").name
                id_path = mask_id_dir / video_dir / output_name
                color_path = mask_color_dir / video_dir / output_name
                overlay_suffix = ".jpg" if args.overlay_format == "jpg" else ".png"
                overlay_path = overlay_dir / video_dir / Path(name).with_suffix(
                    overlay_suffix
                ).name
                for parent in (id_path.parent, color_path.parent, overlay_path.parent):
                    parent.mkdir(parents=True, exist_ok=True)

                Image.fromarray(prediction_original, mode="L").save(id_path)
                Image.fromarray(color_mask, mode="RGB").save(color_path)
                overlay_image = Image.fromarray(blended, mode="RGB")
                if args.overlay_format == "jpg":
                    overlay_image.save(
                        overlay_path,
                        quality=args.jpeg_quality,
                        subsampling=0,
                    )
                else:
                    overlay_image.save(overlay_path)

                manifest_rows.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "source": str(source_path),
                        "mask_id": str(id_path),
                        "mask_color": str(color_path),
                        "overlay": str(overlay_path),
                        "predicted_classes": ",".join(
                            str(int(value)) for value in np.unique(prediction_original)
                        ),
                    }
                )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "video_id",
            "frame_id",
            "source",
            "mask_id",
            "mask_color",
            "overlay",
            "predicted_classes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} predictions")
    print(f"Class-ID masks : {mask_id_dir}")
    print(f"Color masks    : {mask_color_dir}")
    print(f"Overlays       : {overlay_dir}")
    print(f"Manifest       : {manifest_path}")


if __name__ == "__main__":
    main()
