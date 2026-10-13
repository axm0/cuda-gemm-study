# Measurement protocol

Every number in the README comes from this procedure, so it can be rerun.

1. Fresh process per run. `cudaEvent` timing around `iters` back-to-back launches after one warm-up launch; report the mean. Default `iters=20`.
2. Correctness first. Each stage's output is compared elementwise to cuBLAS on the same inputs; `max_rel_err` must be below 1e-3 or the row is marked WRONG and not counted.
3. Square fp32 problems at 1024, 2048, 4096. Stage 0 is skipped above 2048 because it takes minutes and tells us nothing new.
4. Percent of cuBLAS is GFLOP/s of the stage divided by GFLOP/s of `cublasSgemm` at the same size on the same run. cuBLAS is the ceiling, not a competitor.
5. Nsight Compute (`scripts/profile.sh`) captures, per stage at 2048: SM throughput, DRAM throughput, L1 and L2 bytes, achieved occupancy, long-scoreboard and MIO-throttle stall ratios, registers per thread, static smem per block, FFMA count. These are the numbers that explain *why* each stage moved.
6. Clock behaviour is not pinned. On a shared cloud GPU, run the bench twice and take the second; report the GPU model and driver from `nvidia-smi`.

What each stage is meant to show in the profile:

| Stage | Change | Expect in ncu |
|---|---|---|
| 0 naive | row on threadIdx.x | low DRAM efficiency, sectors/request high, long-scoreboard stalls dominate |
| 1 coalesced | col on threadIdx.x | DRAM throughput up, same FFMA count, still memory bound |
| 2 smem tiled | 32x32 tile reuse | L2 bytes drop ~T times, occupancy limited by smem, MIO stalls appear |
| 3 reg tiled | 8x8 micro-tile per thread | FFMA per load up, registers/thread up, occupancy down but SM throughput up |
| 4 vectorized | float4 loads/stores | fewer load instructions, L1 wavefronts down, small but measurable gain |
| 5 cuBLAS | vendor | the ceiling; typically tensor-core-free SGEMM on these GPUs still 1.5 to 3x stage 4 |
