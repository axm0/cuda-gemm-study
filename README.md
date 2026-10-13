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

_Filled in from `results/bench.csv` after the first run. GPU, driver, and per-stage GFLOP/s, % of cuBLAS, and the Nsight metrics that explain each jump._

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

Scaffolded and compiles against the CUDA 12 toolchain in CI (see `.github/workflows/build.yml`). First GPU run pending; results section will be filled from that run only, never estimated.
