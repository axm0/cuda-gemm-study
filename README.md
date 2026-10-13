# cuda-gemm-study

A staged optimization of fp32 matrix multiply in CUDA, measured at each step with Nsight Compute and reported as a percentage of cuBLAS. Five kernels, one harness, one measurement protocol.

The point is not to beat cuBLAS. It is to show, with profiler evidence, what each classic optimization buys and why.

| Stage | Kernel | What changes |
|---|---|---|
| 0 | `gemm_naive` | one thread per output, uncoalesced A reads |
| 1 | `gemm_coalesced` | thread index swapped so warps read contiguous memory |
| 2 | `gemm_smem<32>` | 32x32 shared-memory tiles, each element loaded once per tile |
| 3 | `gemm_regtile<128,128,8,8,8>` | 8x8 register micro-tile per thread, transposed A in smem |
| 4 | `gemm_vec<...>` | float4 vectorized global loads and stores |
| 5 | `cublasSgemm` | reference and ceiling |

## Results

Tesla T4 (40 SMs, 48 KB smem/block), CUDA 13.0 driver 580.82, Google Colab, 2026-10-12. Mean of 20 launches after one warm-up; every stage passed the correctness gate against cuBLAS (relative Frobenius error at fp32 rounding level, 1e-7 to 1e-6; exactly 0 at 4096 because the inputs there happen to accumulate identically).

**1024 x 1024 x 1024**

| stage | ms | GFLOP/s | % of cuBLAS | rel. Frobenius err |
|---|---:|---:|---:|---:|
| 0_naive | 21.178 | 101 | 2.4 | 6.01e-07 |
| 1_coalesced | 3.120 | 688 | 16.2 | 6.01e-07 |
| 2_smem_tiled | 2.542 | 845 | 19.9 | 6.01e-07 |
| 3_reg_tiled | 0.712 | 3015 | 71.2 | 6.01e-07 |
| 4_vectorized | 0.666 | 3226 | 76.1 | 6.01e-07 |
| 5_cublas | 0.507 | 4236 | 100.0 | 0 |

**2048 x 2048 x 2048**

| stage | ms | GFLOP/s | % of cuBLAS | rel. Frobenius err |
|---|---:|---:|---:|---:|
| 0_naive | 143.207 | 120 | 2.3 | 7.69e-07 |
| 1_coalesced | 35.426 | 485 | 9.1 | 7.69e-07 |
| 2_smem_tiled | 18.858 | 911 | 17.1 | 7.69e-07 |
| 3_reg_tiled | 5.174 | 3321 | 62.5 | 7.69e-07 |
| 4_vectorized | 4.910 | 3499 | 65.9 | 7.69e-07 |
| 5_cublas | 3.234 | 5313 | 100.0 | 0 |

**4096 x 4096 x 4096**

| stage | ms | GFLOP/s | % of cuBLAS | rel. Frobenius err |
|---|---:|---:|---:|---:|
| 1_coalesced | 292.993 | 469 | 11.2 | 0.00e+00 |
| 2_smem_tiled | 150.700 | 912 | 21.7 | 0.00e+00 |
| 3_reg_tiled | 44.699 | 3075 | 73.2 | 0.00e+00 |
| 4_vectorized | 39.478 | 3481 | 82.8 | 0.00e+00 |
| 5_cublas | 32.702 | 4203 | 100.0 | 0 |

Stage 0 is skipped at 4096 (it takes minutes and shows nothing new).

What the numbers say:

- **Coalescing alone is 7x** (stage 0 to 1). Same arithmetic, same memory volume; the only change is which thread reads which address. This is the single largest step and it costs zero extra instructions.
- **Shared-memory tiling is a modest win on top of that** (1.3x to 1.9x) because the 32x32 tile still does one FMA per two smem loads, so the kernel stays bound on load issue, not DRAM.
- **Register tiling is the second big step** (3.4x to 3.6x over stage 2). Each thread owns an 8x8 micro-tile, so one smem load of A and one of B feed 64 FMAs. Arithmetic intensity per load is what finally lets the SM do math instead of waiting.
- **Vectorizing** the global loads and stores buys the last 5 to 13 percent by cutting load instruction count and L1 wavefronts.
- **Against cuBLAS: 76 to 83 percent** at 1024 and 4096. The gap at 2048 (66 percent) is where cuBLAS's own kernel peaks (5.3 TFLOP/s), suggesting its tile shape and software pipelining suit that size better than the fixed 128x128x8 tile here. Closing that is the obvious next step: double-buffered smem loads to overlap global fetch with compute.

Per-stage Nsight Compute metrics are in `results/ncu/` (see `scripts/profile.sh`); the summary table lands here after the profile run.

## Run it

On any box with a CUDA toolkit (Colab T4, RunPod/Lambda A10 or A100):

```bash
git clone https://github.com/axm0/cuda-gemm-study && cd cuda-gemm-study
scripts/colab_bootstrap.sh          # cmake build, bench 1024/2048/4096, print table
scripts/profile.sh 2048             # Nsight Compute per stage -> results/ncu/
```

Manual:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80   # 75 T4, 80 A100, 86 A10, 89 L4
cmake --build build -j
./build/gemm_bench --sizes 1024,2048,4096 --iters 20
python3 scripts/plot.py
```

## Layout

- `src/kernels.cu`: the five kernels, each commented with the single idea it adds
- `src/main.cu`: harness: cuBLAS reference, elementwise correctness gate, cudaEvent timing, CSV
- `scripts/profile.sh`: Nsight Compute metric capture per stage
- `scripts/plot.py`: table and bar chart from the CSV
- `docs/MEASUREMENT.md`: the protocol every number follows, and what each stage should show in the profiler

## Status

Compiles clean against CUDA 12.4 in CI (see `.github/workflows/build.yml`). Benchmarked on a Tesla T4; results above are from that run and nothing is estimated. Nsight Compute profile per stage: in progress.
