from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    cpu = load_json("outputs/cpu_benchmark.json")
    gpu = load_json("outputs/gpu_benchmark.json")

    summary = {
        "cpu_available": cpu is not None,
        "gpu_available": gpu is not None,
        "cpu_p95_latency_ms": cpu.get("p95_latency_ms") if cpu else None,
        "gpu_p95_latency_ms": gpu.get("p95_latency_ms") if gpu else None,
        "gpu_name": gpu.get("gpu_metadata", {}).get("gpu_name") if gpu else None,
        "speedup_p95": None,
    }

    if cpu and gpu and gpu.get("p95_latency_ms", 0) > 0:
        summary["speedup_p95"] = round(cpu["p95_latency_ms"] / gpu["p95_latency_ms"], 2)

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
