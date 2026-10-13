// gemm_bench: correctness + timing for each kernel stage against cuBLAS.
// Usage: ./gemm_bench [--sizes 1024,2048,4096] [--iters 20] [--csv results/bench.csv] [--stage N]
#include "kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
#define CB(x) do { cublasStatus_t e = (x); if (e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS err %d at %s:%d\n", (int)e, __FILE__, __LINE__); exit(1);} } while (0)

static void cublas_gemm(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N, int K) {
  // Row-major C = A*B  ==  column-major C^T = B^T * A^T
  const float alpha = 1.f, beta = 0.f;
  CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

// Correctness: relative Frobenius error ||C - Cref||_F / ||Cref||_F, plus max absolute error.
// A per-element relative error is the wrong metric here: elements of C that happen to sit near
// zero turn ordinary fp32 summation-order differences into huge ratios.
struct Err { double rel_fro; double max_abs; };
static Err compare(const std::vector<float>& ref, const std::vector<float>& got) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = (double)ref[i] - got[i];
    num += d * d; den += (double)ref[i] * ref[i]; mx = fmax(mx, fabs(d));
  }
  return { sqrt(num) / (sqrt(den) + 1e-30), mx };
}
static const double REL_FRO_TOL = 1e-4;  // fp32 reorder of K products lands around 1e-6; 1e-4 is generous

int main(int argc, char** argv) {
  std::vector<int> sizes = {1024, 2048, 4096};
  int iters = 20; std::string csv = "results/bench.csv"; int only = -1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--sizes")) { sizes.clear(); char* t = strtok(argv[++i], ","); while (t) { sizes.push_back(atoi(t)); t = strtok(nullptr, ","); } }
    else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!strcmp(argv[i], "--stage")) only = atoi(argv[++i]);
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s  SMs=%d  smem/block=%zuKB\n", prop.name, prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);

  cublasHandle_t h; CB(cublasCreate(&h));
  FILE* f = fopen(csv.c_str(), "w");
  fprintf(f, "gpu,stage,M,N,K,ms,gflops,pct_of_cublas,rel_fro_err,max_abs_err\n");

  std::mt19937 rng(42); std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (int n : sizes) {
    int M = n, N = n, K = n;
    size_t sa = (size_t)M * K, sb = (size_t)K * N, sc = (size_t)M * N;
    std::vector<float> hA(sa), hB(sb), hC(sc), hRef(sc);
    for (auto& x : hA) x = U(rng); for (auto& x : hB) x = U(rng);
    float *dA, *dB, *dC; CK(cudaMalloc(&dA, sa * 4)); CK(cudaMalloc(&dB, sb * 4)); CK(cudaMalloc(&dC, sc * 4));
    CK(cudaMemcpy(dA, hA.data(), sa * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sb * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    double flop = 2.0 * M * N * K;

    // cuBLAS reference + timing
    cublas_gemm(h, dA, dB, dC, M, N, K); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hRef.data(), dC, sc * 4, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e0)); for (int i = 0; i < iters; ++i) cublas_gemm(h, dA, dB, dC, M, N, K); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms_cb; CK(cudaEventElapsedTime(&ms_cb, e0, e1)); ms_cb /= iters;
    double gf_cb = flop / (ms_cb * 1e6);
    printf("\n[%d x %d x %d]\n  %-14s %8.3f ms  %8.1f GFLOP/s  (100.0%%)\n", M, N, K, "5_cublas", ms_cb, gf_cb);
    fprintf(f, "%s,5_cublas,%d,%d,%d,%.4f,%.1f,100.0,0,0\n", prop.name, M, N, K, ms_cb, gf_cb);

    for (int s = 0; s < 5; ++s) {
      if (only >= 0 && s != only) continue;
      Stage st = (Stage)s;
      if (s == 0 && n > 2048) { printf("  %-14s skipped (too slow at this size)\n", stage_name(st)); continue; }
      CK(cudaMemset(dC, 0, sc * 4));
      launch(st, dA, dB, dC, M, N, K, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(hC.data(), dC, sc * 4, cudaMemcpyDeviceToHost));
      Err err = compare(hRef, hC);
      CK(cudaEventRecord(e0)); for (int i = 0; i < iters; ++i) launch(st, dA, dB, dC, M, N, K, 0); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= iters;
      double gf = flop / (ms * 1e6), pct = 100.0 * gf / gf_cb;
      printf("  %-14s %8.3f ms  %8.1f GFLOP/s  (%5.1f%%)  rel_fro=%.1e max_abs=%.1e%s\n", stage_name(st), ms, gf, pct, err.rel_fro, err.max_abs, err.rel_fro < REL_FRO_TOL ? "" : "  <-- WRONG");
      fprintf(f, "%s,%s,%d,%d,%d,%.4f,%.1f,%.1f,%.2e,%.2e\n", prop.name, stage_name(st), M, N, K, ms, gf, pct, err.rel_fro, err.max_abs);
    }
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
  }
  fclose(f); CB(cublasDestroy(h));
  printf("\nwrote %s\n", csv.c_str());
  return 0;
}
