from __future__ import annotations

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
import omni.replicator.core as rep


USD_PATH = Path("outputs/isaac_healthcare_or_digital_twin.usd").resolve()
OUT_DIR = Path("/root/omni.replicator_out/isaac_scene_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Opening USD scene: {USD_PATH}")
omni.usd.get_context().open_stage(str(USD_PATH))

# Give Isaac time to load the stage.
for _ in range(60):
    simulation_app.update()

print("Creating camera, lights, and render product...")

with rep.new_layer():
    # Strong lighting so the scene is not black.
    rep.create.light(
        light_type="Dome",
        intensity=2500,
        color=(1.0, 1.0, 1.0),
    )

    rep.create.light(
        light_type="Sphere",
        position=(0, -2.5, 4.0),
        intensity=8000,
        color=(1.0, 1.0, 1.0),
    )

    # Camera aimed at patient table / scene center.
    camera = rep.create.camera(
        position=(3.5, -3.5, 2.6),
        look_at=(0.0, 0.0, 0.75),
        focal_length=24,
    )

    render_product = rep.create.render_product(camera, (1280, 720))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(OUT_DIR),
        rgb=True,
    )
    writer.attach([render_product])

    print("Rendering frames...")
    for _ in range(5):
        rep.orchestrator.step()
        simulation_app.update()

    rep.orchestrator.wait_until_complete()

for _ in range(20):
    simulation_app.update()

print(f"Rendered preview written under: {OUT_DIR}")
simulation_app.close()
