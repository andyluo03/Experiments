#include "fast_mm.hpp"

namespace {
__global__
void naive_matrix_multiplication (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
) {
    int outRow = blockIdx.x * blockDim.x + threadIdx.x;
    int outCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (outRow < M && outCol < K) {
        float result = 0.0f;

        for (int i = 0; i < N; i++) {
            result += A[outRow * N + i] * B[i * K + outCol];
        }

        C[outRow * K + outCol] = result;
    }
}
}

void naive_gemm_32 (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
) {
    dim3 blockDim(32, 32);
    dim3 gridDim((M + 31)/32, (K + 31)/32);

    naive_matrix_multiplication<<<gridDim, blockDim>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}
