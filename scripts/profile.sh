#!/usr/bin/env bash
# Profile every stage at one size with Nsight Compute and dump the metrics the write-up uses.
# Usage: scripts/profile.sh [size]   (default 2048). Needs ncu on PATH (ships with CUDA toolkit).
set -euo pipefail
SIZE="${1:-2048}"
mkdir -p results/ncu
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes.sum,lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,launch__registers_per_thread,launch__shared_mem_per_block_static,sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
for s in 0 1 2 3 4; do
  echo "== stage $s"
  ncu --metrics "$METRICS" --csv --kernel-name regex:gemm_ --launch-skip 1 --launch-count 1 \
      ./build/gemm_bench --sizes "$SIZE" --iters 2 --stage "$s" --csv /dev/null \
      > "results/ncu/stage${s}_${SIZE}.csv" 2>/dev/null || echo "  ncu failed for stage $s (is ncu on PATH? did you run as root / with --target-processes all?)"
done
echo "wrote results/ncu/*.csv"
