#include "kernels.cuh"
#include <cuda_runtime.h>

const char* stage_name(Stage s) {
  switch (s) {
    case Stage::Naive: return "0_naive";
    case Stage::Coalesced: return "1_coalesced";
    case Stage::SmemTiled: return "2_smem_tiled";
    case Stage::RegTiled: return "3_reg_tiled";
    case Stage::Vectorized: return "4_vectorized";
    case Stage::Cublas: return "5_cublas";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Stage 0: one thread per output element, row index on threadIdx.x.
// Deliberately bad: consecutive threads read A along a column (stride K),
// so global loads are uncoalesced. This is the baseline Nsight will flag.
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 1: same math, but col on threadIdx.x so a warp reads 32 consecutive
// floats of B and writes 32 consecutive floats of C. Fixes coalescing only.
// ---------------------------------------------------------------------------
__global__ void gemm_coalesced(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 2: shared-memory tiling. Each block loads a TxT tile of A and B into
// smem once, then every thread reuses it T times. Cuts global traffic by ~T.
// ---------------------------------------------------------------------------
template <int T>
__global__ void gemm_smem(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (col < N && k0 + ty < K) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 3: register tiling. Each thread owns a TMxTN micro-tile of C held in
// registers, raising arithmetic intensity per smem load. Block tile BMxBN,
// K-slice BK. Thread count = (BM/TM)*(BN/TN).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_regtile(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM];  // transposed so the inner loop reads As[k][row] contiguously
  __shared__ float Bs[BK][BN];
  int tid = threadIdx.x;
  int trow = tid / (BN / TN);
  int tcol = tid % (BN / TN);
  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // cooperative load: each thread loads (BM*BK)/THREADS elements of A and (BK*BN)/THREADS of B
    for (int i = tid; i < BM * BK; i += THREADS) {
      int r = i / BK, c = i % BK;
      int gr = block_row + r, gc = k0 + c;
      As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      int r = i / BN, c = i % BN;
      int gr = k0 + r, gc = block_col + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[k][trow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tcol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gr = block_row + trow * TM + i;
    if (gr >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gc = block_col + tcol * TN + j;
      if (gc < N) C[gr * N + gc] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 4: same as stage 3 plus float4 vectorized global loads/stores.
// Requires K % 4 == 0 and N % 4 == 0 (the harness pads sizes to multiples of 128).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_vec(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  int tid = threadIdx.x;
  int trow = tid / (BN / TN);
  int tcol = tid % (BN / TN);
  int block_row = blockIdx.y * BM;
  int block_col = blockIdx.x * BN;
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid * 4; i < BM * BK; i += THREADS * 4) {
      int r = i / BK, c = i % BK;
      float4 v = *reinterpret_cast<const float4*>(&A[(block_row + r) * K + k0 + c]);
      As[c + 0][r] = v.x; As[c + 1][r] = v.y; As[c + 2][r] = v.z; As[c + 3][r] = v.w;
    }
    for (int i = tid * 4; i < BK * BN; i += THREADS * 4) {
      int r = i / BN, c = i % BN;
      *reinterpret_cast<float4*>(&Bs[r][c]) =
          *reinterpret_cast<const float4*>(&B[(k0 + r) * N + block_col + c]);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[k][trow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tcol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gr = block_row + trow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      int gc = block_col + tcol * TN + j;
      *reinterpret_cast<float4*>(&C[gr * N + gc]) = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
    }
  }
}

void launch(Stage s, const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t st) {
  switch (s) {
    case Stage::Naive: {
      dim3 b(16, 16), g((M + 15) / 16, (N + 15) / 16);
      gemm_naive<<<g, b, 0, st>>>(A, B, C, M, N, K); break;
    }
    case Stage::Coalesced: {
      dim3 b(32, 8), g((N + 31) / 32, (M + 7) / 8);
      gemm_coalesced<<<g, b, 0, st>>>(A, B, C, M, N, K); break;
    }
    case Stage::SmemTiled: {
      constexpr int T = 32;
      dim3 b(T, T), g((N + T - 1) / T, (M + T - 1) / T);
      gemm_smem<T><<<g, b, 0, st>>>(A, B, C, M, N, K); break;
    }
    case Stage::RegTiled: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      dim3 b((BM / TM) * (BN / TN)), g((N + BN - 1) / BN, (M + BM - 1) / BM);
      gemm_regtile<BM, BN, BK, TM, TN><<<g, b, 0, st>>>(A, B, C, M, N, K); break;
    }
    case Stage::Vectorized: {
      constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
      dim3 b((BM / TM) * (BN / TN)), g((N + BN - 1) / BN, (M + BM - 1) / BM);
      gemm_vec<BM, BN, BK, TM, TN><<<g, b, 0, st>>>(A, B, C, M, N, K); break;
    }
    case Stage::Cublas: break;  // handled in main.cu
  }
}
