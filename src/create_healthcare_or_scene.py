from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": 1280,
        "height": 720,
    }
)

import omni.usd
from pxr import Gf, UsdGeom, UsdLux


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def add_cube(stage, path: str, translate, scale, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube.AddScaleOp().Set(Gf.Vec3f(*scale))

    prim = cube.GetPrim()
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    return cube


def add_sphere(stage, path: str, translate, scale, color):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*translate))
    sphere.AddScaleOp().Set(Gf.Vec3f(*scale))

    prim = sphere.GetPrim()
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    return sphere


def main():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, "/World")

    add_cube(
        stage,
        "/World/OR_Floor",
        translate=(0, 0, -0.05),
        scale=(5.0, 4.0, 0.05),
        color=(0.55, 0.55, 0.55),
    )

    add_cube(
        stage,
        "/World/Patient_Table",
        translate=(0, 0, 0.65),
        scale=(1.4, 0.45, 0.08),
        color=(0.20, 0.45, 0.75),
    )

    add_sphere(
        stage,
        "/World/Patient_Phantom",
        translate=(0, 0, 0.90),
        scale=(0.65, 0.28, 0.18),
        color=(0.95, 0.75, 0.65),
    )

    add_cube(
        stage,
        "/World/CArm_Base",
        translate=(-1.5, 0.0, 0.25),
        scale=(0.25, 0.35, 0.25),
        color=(0.10, 0.10, 0.10),
    )

    add_cube(
        stage,
        "/World/CArm_Vertical_Support",
        translate=(-1.5, 0.0, 1.0),
        scale=(0.08, 0.08, 0.8),
        color=(0.85, 0.85, 0.85),
    )

    add_cube(
        stage,
        "/World/CArm_Detector",
        translate=(-0.65, 0.0, 1.35),
        scale=(0.22, 0.10, 0.22),
        color=(0.05, 0.05, 0.05),
    )

    add_cube(
        stage,
        "/World/CArm_Source",
        translate=(-0.65, 0.0, 0.55),
        scale=(0.18, 0.10, 0.18),
        color=(0.15, 0.15, 0.15),
    )

    add_cube(
        stage,
        "/World/Robot_Cart_Base",
        translate=(1.6, -0.8, 0.25),
        scale=(0.30, 0.25, 0.20),
        color=(0.05, 0.30, 0.15),
    )

    add_cube(
        stage,
        "/World/Robot_Arm_Link1",
        translate=(1.35, -0.45, 0.75),
        scale=(0.08, 0.08, 0.55),
        color=(0.10, 0.60, 0.25),
    )

    add_cube(
        stage,
        "/World/Robot_Arm_Link2",
        translate=(1.05, -0.20, 1.15),
        scale=(0.45, 0.06, 0.06),
        color=(0.10, 0.60, 0.25),
    )

    add_cube(
        stage,
        "/World/Clinical_Monitor",
        translate=(1.5, 1.1, 1.3),
        scale=(0.45, 0.05, 0.28),
        color=(0.02, 0.02, 0.02),
    )

    light = UsdLux.DistantLight.Define(stage, "/World/OR_Overhead_Light")
    light.CreateIntensityAttr(700)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 45))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(3.0, -3.0, 2.4))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(60, 0, 42))

    usd_path = OUT_DIR / "isaac_healthcare_or_digital_twin.usd"
    context.save_as_stage(str(usd_path))

    metadata = {
        "scene_name": "isaac_healthcare_or_digital_twin",
        "usd_file": str(usd_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_elements": [
            "OR floor",
            "patient table",
            "patient phantom",
            "C-arm imaging device placeholder",
            "robot/cart placeholder",
            "clinical monitor",
            "overhead light",
            "camera",
        ],
        "purpose": "Lightweight Isaac Sim healthcare digital-twin scene for image-guided intervention / physical-AI workflow discussion.",
        "disclaimer": "Conceptual scene only. Not a clinical simulation or validated robotic model.",
    }

    metadata_path = OUT_DIR / "isaac_scene_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"Saved Isaac Sim scene to: {usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
