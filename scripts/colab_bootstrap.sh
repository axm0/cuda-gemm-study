#!/usr/bin/env bash
# One-shot setup on a fresh Colab / RunPod / Lambda box with a CUDA toolkit already present.
set -euo pipefail
nvidia-smi | head -15
command -v cmake >/dev/null || (apt-get update -qq && apt-get install -y -qq cmake >/dev/null)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j
./build/gemm_bench --sizes 1024,2048,4096 --iters 20
python3 scripts/plot.py
