from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from infer_edge_workflow import load_case, load_model
from model import get_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default="case_000")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--model-path", type=str, default="outputs/model.pt")
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    args = parser.parse_args()

    device = get_device()

    x, _ = load_case(args.case_id, args.data_dir)
    model, ckpt = load_model(args.model_path, device)

    x = x.to(device)

    for _ in range(args.warmup_runs):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    for _ in range(args.num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(x)
        _ = torch.argmax(logits, dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies.append((time.perf_counter() - start) * 1000)

    latencies_np = np.array(latencies, dtype=np.float64)

    result = {
        "workflow": "synthetic_mri_ct_to_autocontour_latency_benchmark",
        "case_id": args.case_id,
        "model_name": ckpt.get("model_name", "monai_toy_autocontour"),
        "model_version": ckpt.get("model_version", "0.1.0"),
        "device": str(device),
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "mean_latency_ms": round(float(np.mean(latencies_np)), 3),
        "p50_latency_ms": round(float(np.percentile(latencies_np, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_np, 95)), 3),
        "min_latency_ms": round(float(np.min(latencies_np)), 3),
        "max_latency_ms": round(float(np.max(latencies_np)), 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Toy synthetic demo. Not clinical-grade benchmark.",
    }

    out_file = "outputs/gpu_benchmark.json" if device.type == "cuda" else "outputs/cpu_benchmark.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved benchmark to {out_file}")


if __name__ == "__main__":
    main()