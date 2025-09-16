#pragma once

void tiled_gemm_32 (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
);