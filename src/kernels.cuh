#pragma once
#include <cuda_runtime.h>
// All kernels compute C = A * B for row-major MxK A, KxN B, MxN C (fp32).
// Each stage is one optimization step; the write-up measures each with Nsight Compute.
enum class Stage { Naive = 0, Coalesced, SmemTiled, RegTiled, Vectorized, Cublas };
const char* stage_name(Stage s);
void launch(Stage s, const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
