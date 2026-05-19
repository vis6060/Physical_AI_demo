from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_model, get_device


def load_case(case_id: str, data_dir: str = "data/synthetic") -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    path = Path(data_dir) / f"{case_id}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}. Run generate_synthetic_mri_ct.py first.")

    data = np.load(path)
    x = np.stack([data["mri"], data["ct"]], axis=0).astype(np.float32)

    return torch.from_numpy(x).unsqueeze(0), {k: data[k] for k in data.files}


def load_model(model_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt_path = Path(model_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_monai_autocontour.py first.")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(
        in_channels=ckpt.get("input_channels", 2),
        out_channels=ckpt.get("output_classes", 3),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def save_overlay(mri: np.ndarray, mask_pred: np.ndarray, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(mri, cmap="gray")

    overlay = np.ma.masked_where(mask_pred == 0, mask_pred)
    plt.imshow(overlay, cmap="autumn", alpha=0.45)

    plt.axis("off")
    plt.title("Predicted auto-contour overlay")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default="case_000")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--model-path", type=str, default="outputs/model.pt")
    parser.add_argument("--out-pred", type=str, default="outputs/sample_prediction.npy")
    parser.add_argument("--out-overlay", type=str, default="outputs/sample_prediction_overlay.png")
    parser.add_argument("--audit-log", type=str, default="outputs/edge_audit_log.json")
    args = parser.parse_args()

    device = get_device()

    x, case = load_case(args.case_id, args.data_dir)
    model, ckpt = load_model(args.model_path, device)

    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    logits = model(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    np.save(args.out_pred, pred)
    save_overlay(case["mri"], pred, args.out_overlay)

    audit = {
        "workflow": "synthetic_mri_ct_to_autocontour",
        "case_id": args.case_id,
        "model_name": ckpt.get("model_name", "monai_toy_autocontour"),
        "model_version": ckpt.get("model_version", "0.1.0"),
        "device": str(device),
        "latency_ms": round(elapsed_ms, 3),
        "input_channels": ["synthetic_mri", "synthetic_ct"],
        "output_classes": {"0": "background", "1": "organ", "2": "tumor"},
        "prediction_file": args.out_pred,
        "overlay_file": args.out_overlay,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Toy synthetic demo. Not patient data. Not clinical-grade.",
    }

    Path(args.audit_log).parent.mkdir(parents=True, exist_ok=True)

    with open(args.audit_log, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()