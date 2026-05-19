from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Healthcare Physical-AI Edge Inference MVP")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "healthcare-physical-ai-edge-mvp",
    }


@app.post("/infer/{case_id}")
def infer(case_id: str) -> dict:
    cmd = [
        sys.executable,
        "src/infer_edge_workflow.py",
        "--case-id",
        case_id,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr or str(e))

    log_path = Path("outputs/edge_audit_log.json")

    if not log_path.exists():
        raise HTTPException(status_code=500, detail="Audit log not created")

    return json.loads(log_path.read_text(encoding="utf-8"))