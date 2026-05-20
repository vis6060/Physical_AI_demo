
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.conditions import CountCondition

from model import build_model, get_device


class DataSourceOp(Operator):
    def __init__(self, fragment, *args, case_id: str, data_dir: str, **kwargs):
        self.case_id = case_id
        self.data_dir = Path(data_dir)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        case_path = self.data_dir / f"{self.case_id}.npz"
        if not case_path.exists():
            raise FileNotFoundError(f"Missing case: {case_path}")

        data = np.load(case_path)

        message = {
            "case_id": self.case_id,
            "mri": data["mri"].astype(np.float32),
            "ct": data["ct"].astype(np.float32),
            "ground_truth_mask": data["mask"].astype(np.int64),
        }

        op_output.emit(message, "out")


class PreprocessOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")

        x = np.stack([message["mri"], message["ct"]], axis=0).astype(np.float32)
        x_tensor = torch.from_numpy(x).unsqueeze(0)

        message["input_tensor"] = x_tensor
        op_output.emit(message, "out")


class MONAIInferenceOp(Operator):
    def __init__(self, fragment, *args, model_path: str, **kwargs):
        self.model_path = Path(model_path)
        self.device = get_device()
        self.model = None
        self.ckpt = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def start(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model: {self.model_path}")

        self.ckpt = torch.load(self.model_path, map_location=self.device)

        self.model = build_model(
            in_channels=self.ckpt.get("input_channels", 2),
            out_channels=self.ckpt.get("output_classes", 3),
        ).to(self.device)

        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")

        x = message["input_tensor"].to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = self.model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        message["prediction"] = pred
        message["device"] = str(self.device)
        message["latency_ms"] = round(latency_ms, 3)
        message["model_name"] = self.ckpt.get("model_name", "monai_toy_autocontour")
        message["model_version"] = self.ckpt.get("model_version", "0.1.0")

        op_output.emit(message, "out")


class PostprocessOp(Operator):
    def __init__(self, fragment, *args, out_dir: str, **kwargs):
        self.out_dir = Path(out_dir)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pred = message["prediction"]
        mri = message["mri"]

        pred_path = self.out_dir / "holoscan_prediction.npy"
        overlay_path = self.out_dir / "holoscan_pipeline_output.png"

        np.save(pred_path, pred)

        plt.figure(figsize=(5, 5))
        plt.imshow(mri, cmap="gray")
        overlay = np.ma.masked_where(pred == 0, pred)
        plt.imshow(overlay, cmap="autumn", alpha=0.45)
        plt.title("Holoscan Pipeline Predicted Contour")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150)
        plt.close()

        message["prediction_file"] = str(pred_path)
        message["overlay_file"] = str(overlay_path)

        op_output.emit(message, "out")


class AuditLogOp(Operator):
    def __init__(self, fragment, *args, out_dir: str, **kwargs):
        self.out_dir = Path(out_dir)
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("in")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        audit = {
            "workflow": "holoscan_synthetic_mri_ct_to_autocontour",
            "case_id": message["case_id"],
            "model_name": message["model_name"],
            "model_version": message["model_version"],
            "device": message["device"],
            "latency_ms": message["latency_ms"],
            "input_channels": ["synthetic_mri", "synthetic_ct"],
            "output_classes": {"0": "background", "1": "organ", "2": "tumor"},
            "prediction_file": message["prediction_file"],
            "overlay_file": message["overlay_file"],
            "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": [
                "DataSourceOp",
                "PreprocessOp",
                "MONAIInferenceOp",
                "PostprocessOp",
                "AuditLogOp",
            ],
            "disclaimer": "Toy synthetic demo. Not patient data. Not clinical-grade.",
        }

        audit_path = self.out_dir / "holoscan_audit_log.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        print(json.dumps(audit, indent=2))


class HealthcarePhysicalAIHoloscanApp(Application):
    def __init__(self, case_id: str, data_dir: str, model_path: str, out_dir: str):
        self.case_id = case_id
        self.data_dir = data_dir
        self.model_path = model_path
        self.out_dir = out_dir
        super().__init__()

    def compose(self):
        source = DataSourceOp(
            self,
            CountCondition(self, 1),
            name="data_source",
            case_id=self.case_id,
            data_dir=self.data_dir,
        )

        preprocess = PreprocessOp(
            self,
            name="preprocess",
        )

        inference = MONAIInferenceOp(
            self,
            name="monai_inference",
            model_path=self.model_path,
        )

        postprocess = PostprocessOp(
            self,
            name="postprocess",
            out_dir=self.out_dir,
        )

        audit = AuditLogOp(
            self,
            name="audit_log",
            out_dir=self.out_dir,
        )

        self.add_flow(source, preprocess)
        self.add_flow(preprocess, inference)
        self.add_flow(inference, postprocess)
        self.add_flow(postprocess, audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default="case_000")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--model-path", type=str, default="outputs/model.pt")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    app = HealthcarePhysicalAIHoloscanApp(
        case_id=args.case_id,
        data_dir=args.data_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
    )

    app.run()


if __name__ == "__main__":
    main()
