from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def make_ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, angle: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    x0 = x - cx
    y0 = y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = ca * x0 + sa * y0
    yr = -sa * x0 + ca * y0
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def generate_case(image_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    h = w = image_size

    body = make_ellipse_mask(
        h, w,
        cx=w / 2 + rng.normal(0, 2),
        cy=h / 2 + rng.normal(0, 2),
        rx=rng.uniform(0.35, 0.43) * w,
        ry=rng.uniform(0.40, 0.48) * h,
        angle=rng.uniform(-0.10, 0.10),
    )

    organ = make_ellipse_mask(
        h, w,
        cx=w / 2 + rng.normal(0, 4),
        cy=h / 2 + rng.normal(0, 4),
        rx=rng.uniform(0.18, 0.27) * w,
        ry=rng.uniform(0.16, 0.25) * h,
        angle=rng.uniform(-0.6, 0.6),
    ) & body

    organ_indices = np.argwhere(organ)
    if len(organ_indices) == 0:
        ty, tx = h // 2, w // 2
    else:
        ty, tx = organ_indices[rng.integers(0, len(organ_indices))]

    tumor = make_ellipse_mask(
        h, w,
        cx=float(tx),
        cy=float(ty),
        rx=rng.uniform(0.035, 0.065) * w,
        ry=rng.uniform(0.035, 0.065) * h,
        angle=rng.uniform(0, np.pi),
    ) & organ

    mask = np.zeros((h, w), dtype=np.int64)
    mask[organ] = 1
    mask[tumor] = 2

    y, x = np.mgrid[0:h, 0:w]
    bias = 0.85 + 0.25 * (x / max(w - 1, 1)) + 0.10 * np.sin(2 * np.pi * y / h)

    mri = np.zeros((h, w), dtype=np.float32)
    mri[body] = rng.normal(0.35, 0.04, size=body.sum())
    mri[organ] = rng.normal(0.68, 0.06, size=organ.sum())
    mri[tumor] = rng.normal(0.92, 0.05, size=tumor.sum())
    mri = gaussian_filter(mri * bias, sigma=rng.uniform(0.5, 1.2))
    mri += rng.normal(0, rng.uniform(0.015, 0.035), size=(h, w))
    mri = normalize01(mri)

    ct_hu = np.full((h, w), -1000.0, dtype=np.float32)
    ct_hu[body] = rng.normal(35, 20, size=body.sum())
    ct_hu[organ] = rng.normal(65, 15, size=organ.sum())
    ct_hu[tumor] = rng.normal(95, 18, size=tumor.sum())

    for _ in range(rng.integers(1, 4)):
        bx = rng.uniform(0.25, 0.75) * w
        by = rng.uniform(0.25, 0.75) * h
        bone = make_ellipse_mask(
            h, w, bx, by,
            rx=rng.uniform(0.03, 0.06) * w,
            ry=rng.uniform(0.02, 0.05) * h,
            angle=rng.uniform(0, np.pi),
        ) & body & ~organ
        ct_hu[bone] = rng.normal(600, 90, size=bone.sum())

    ct = normalize01(ct_hu)

    return {
        "mri": mri.astype(np.float32),
        "ct": ct.astype(np.float32),
        "ct_hu": ct_hu.astype(np.float32),
        "mask": mask.astype(np.int64),
    }


def save_png(path: Path, arr: np.ndarray, cmap: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/synthetic")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for i in range(args.num_cases):
        case_id = f"case_{i:03d}"
        case = generate_case(args.image_size, rng)
        np.savez_compressed(out_dir / f"{case_id}.npz", **case)

        if i == 0:
            save_png(Path("outputs/sample_mri.png"), case["mri"])
            save_png(Path("outputs/sample_ct.png"), case["ct"])
            save_png(Path("outputs/sample_mask.png"), case["mask"], cmap="viridis")

    metadata = {
        "num_cases": args.num_cases,
        "image_size": args.image_size,
        "seed": args.seed,
        "classes": {"0": "background", "1": "organ", "2": "tumor"},
        "disclaimer": "Toy synthetic data. Not patient data. Not clinical-grade.",
    }

    with open("outputs/synthetic_data_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {args.num_cases} cases in {out_dir}")
    print("Sample images written to outputs/")


if __name__ == "__main__":
    main()