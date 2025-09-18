#include "tiled_mm.hpp"

constexpr int kTileSize = 32;

namespace {
__global__
void tiled_matrix_multiplication (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
) {
    __shared__ float A_tile[kTileSize][kTileSize];
    __shared__ float B_tile[kTileSize][kTileSize];

    int flattened_ind = threadIdx.y * blockDim.x + threadIdx.x;

    int ld_block_A_row = blockIdx.x * kTileSize;
    int ld_local_A_row = flattened_ind / kTileSize;
    int ld_local_A_col = flattened_ind % kTileSize;

    int ld_block_B_col = blockIdx.y * kTileSize;
    int ld_local_B_row = flattened_ind % kTileSize;
    int ld_local_B_col = flattened_ind / kTileSize;

    int out_local_Row = threadIdx.x;
    int out_local_Col = threadIdx.y; 

    int outRow = blockIdx.x * blockDim.x + threadIdx.x;
    int outCol = blockIdx.y * blockDim.y + threadIdx.y;

    if (outRow < M && outCol < K) {
        float result = 0.0f;

        for (int position = 0; position < N + kTileSize - 1; position += kTileSize) {
            A_tile[ld_local_A_row][ld_local_A_col] = A[
                (ld_block_A_row +ld_local_A_row) * N + (position + ld_local_A_col)
            ];

            // note: could reduce smem conflict... but for sake of example, don't.
            B_tile[ld_local_B_col][ld_local_B_row] = B[
                (ld_local_B_row + position) * K + (ld_local_B_col + ld_block_B_col)
            ];

            __syncthreads();

            for (int i = 0; i < kTileSize; i++) {
                result += A_tile[out_local_Row][i] * B_tile[i][out_local_Col];
            }

            __syncthreads();
        }

        C[outRow * K + outCol] += result;
    }
}
}

void tiled_mm_32 (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
) {
    dim3 blockDim(kTileSize, kTileSize);
    dim3 gridDim((M + kTileSize - 1)/kTileSize, (K + kTileSize - 1)/kTileSize);

    tiled_matrix_multiplication<<<gridDim, blockDim>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}
