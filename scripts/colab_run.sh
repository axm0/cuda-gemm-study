#!/usr/bin/env bash
# Rebuild, bench, dump CSV, and attempt an Nsight Compute capture. Safe to re-run.
set -uo pipefail
cmake --build build -j 2>&1 | tail -1
./build/gemm_bench --sizes 1024,2048,4096 --iters 20
echo "===CSV==="; cat results/bench.csv; echo "===ENDCSV==="
echo "===NCU==="
if command -v ncu >/dev/null; then
  ncu --version | tail -1
  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,lts__t_bytes.sum \
      --csv --kernel-name regex:gemm_ --launch-skip 1 --launch-count 1 \
      ./build/gemm_bench --sizes 2048 --iters 2 --stage 3 --csv /dev/null 2>&1 | tail -12
else
  echo "ncu not on PATH"; ls /usr/local/cuda*/bin 2>/dev/null | grep -i ncu || true
fi
echo "===ENDNCU==="
