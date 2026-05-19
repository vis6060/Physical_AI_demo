# NVIDIA Healthcare Physical-AI Demo — MVP

CPU-first proof-of-concept showing:

synthetic MRI/CT-style data → MONAI auto-contouring → edge-style inference → latency benchmark → audit trace

## What works today

- Generates synthetic MRI/CT-style data
- Creates tumor/organ masks
- Trains a small MONAI U-Net
- Runs tumor/organ auto-contouring inference
- Benchmarks CPU inference latency
- Automatically uses CUDA later when available
- Writes audit logs, benchmark JSON, and visualization outputs

## Roadmap

- CUDA benchmark on RTX 4090 / H100
- Holoscan SDK operator pipeline
- Isaac Sim healthcare digital-twin scene

## Install — CPU first

```bash
conda create -n nvidia-healthcare-ai python=3.10 -y
conda activate nvidia-healthcare-ai
pip install -r requirements-cpu.txt