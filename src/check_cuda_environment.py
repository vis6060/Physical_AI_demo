from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> None:
    cuda_available = torch.cuda.is_available()

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [],
    }

    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/cuda_environment.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
