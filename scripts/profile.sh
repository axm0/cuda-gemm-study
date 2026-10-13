#!/usr/bin/env bash
# Profile every stage at one size with Nsight Compute. Usage: scripts/profile.sh [size]  (default 2048)
# The kernel filter is anchored (^gemm_) so cuBLAS's own *sgemm* kernels are excluded.
set -uo pipefail
SIZE="${1:-2048}"
mkdir -p results/ncu
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum,l1tex__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
for s in 0 1 2 3 4; do
  out="results/ncu/stage${s}_${SIZE}.csv"
  ncu --metrics "$METRICS" --csv --kernel-name "regex:^gemm_" --launch-skip 1 --launch-count 1 \
      ./build/gemm_bench --sizes "$SIZE" --iters 2 --stage "$s" --csv /dev/null 2>/dev/null \
      | grep -E '^"' > "$out" || echo "stage $s: ncu failed"
  echo "== stage $s: $(grep -c . "$out") metric rows -> $out"
done
echo "===NCU-SUMMARY==="
python3 - "$SIZE" <<'PY'
import csv,glob,sys
size=sys.argv[1]
want=["sm__throughput.avg.pct_of_peak_sustained_elapsed","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active","launch__registers_per_thread","launch__shared_mem_per_block_static","lts__t_bytes.sum","smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio","smsp__average_warp_latency_issue_stalled_mio_throttle.ratio"]
short=["SM%","DRAM%","occ%","regs","smemB","L2bytes","lsb_stall","mio_stall"]
print("stage,kernel,"+",".join(short))
for s in range(5):
    try: rows=list(csv.DictReader(open(f"results/ncu/stage{s}_{size}.csv")))
    except FileNotFoundError: continue
    if not rows: continue
    k=rows[0]["Kernel Name"][:28]
    m={r["Metric Name"]:r["Metric Value"] for r in rows}
    print(f"{s},{k},"+",".join(m.get(w,"?") for w in want))
PY
echo "===END-NCU-SUMMARY==="
