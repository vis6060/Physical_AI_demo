from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default="case_000")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--pred-file", type=str, default="outputs/sample_prediction.npy")
    parser.add_argument("--out-file", type=str, default="outputs/summary_visualization.png")
    args = parser.parse_args()

    case_path = Path(args.data_dir) / f"{args.case_id}.npz"
    pred_path = Path(args.pred_file)

    if not case_path.exists():
        raise FileNotFoundError(f"Missing case file: {case_path}")

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}. Run infer_edge_workflow.py first.")

    data = np.load(case_path)
    pred = np.load(pred_path)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(data["mri"], cmap="gray")
    axes[0].set_title("Synthetic MRI")

    axes[1].imshow(data["ct"], cmap="gray")
    axes[1].set_title("Synthetic CT")

    axes[2].imshow(data["mask"], cmap="viridis")
    axes[2].set_title("Ground Truth")

    axes[3].imshow(data["mri"], cmap="gray")
    overlay = np.ma.masked_where(pred == 0, pred)
    axes[3].imshow(overlay, cmap="autumn", alpha=0.45)
    axes[3].set_title("Predicted Contour")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_file, dpi=150)
    plt.close()

    print(f"Saved {args.out_file}")


if __name__ == "__main__":
    main()