# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


LEVELS = ("L0", "L1", "L2", "L3")
CLASS_NAMES = (
    "Background",
    "Liver",
    "Gallbladder",
    "Other tissue",
    "Duct",
    "Instrument",
    "Gauze",
    "Hook",
    "Clip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read L0-L3 global confusion matrices and export a per-class "
            "Dice table without rerunning model inference."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing L0_confusion_matrix.csv ... L3_confusion_matrix.csv",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    matrix = np.asarray(
        [[int(value) for value in row[1:]] for row in rows[1:]],
        dtype=np.int64,
    )
    if matrix.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 matrix in {path}, got {matrix.shape}")
    return matrix


def dice_from_matrix(matrix: np.ndarray) -> np.ndarray:
    true_positive = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    per_level = {
        level: dice_from_matrix(
            load_matrix(input_dir / f"{level}_confusion_matrix.csv")
        )
        for level in LEVELS
    }

    csv_path = input_dir / "class_by_smoke_level_dice.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", *LEVELS])
        for class_id, class_name in enumerate(CLASS_NAMES):
            writer.writerow(
                [class_name]
                + [float(per_level[level][class_id]) for level in LEVELS]
            )

    latex_lines = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = " & ".join(
            f"{per_level[level][class_id]:.4f}" for level in LEVELS
        )
        line = f"{class_name} & {values} " + r"\\"
        latex_lines.append(line)
        print(line)

    tex_path = input_dir / "class_by_smoke_level_dice.tex"
    tex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    print(f"CSV saved to: {csv_path}")
    print(f"LaTeX rows saved to: {tex_path}")


if __name__ == "__main__":
    main()
