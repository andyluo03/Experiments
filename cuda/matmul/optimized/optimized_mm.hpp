#pragma once

// Optimizations:
//  * warp specialization (instruction issuing bottleneck)
//      * TMA
//  * tensor cores
//  * tensor memory
//  * swizzling
//  * register pressure/occupancy/etc. tuning

void optimized_mm_32 (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
);