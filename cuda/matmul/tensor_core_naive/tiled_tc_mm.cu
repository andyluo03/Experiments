#include "tiled_tc_mm.hpp"

// PTX
// 16x8 mma
// 4 floats/thread

constexpr int kTileSize = 32;

namespace {

__device__ __forceinline__
void create_C () {

}

__device__ __forceinline__ 
void load_A_from_shared () {

}

__device__ __forceinline__
void load_B_from_shared () {

}

__global__
void tiled_tc_matrix_multiplication (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K 
) {
    __shared__ float A_tile[16][8]; // Single fragment.
    __shared__ float B_tile[8][16];


    // ld.mat.x4

    for () {
        

        __syncthreads();


        for () {
            // LOAD INTO FRAGMENTS

            // MMA into ACCUMULATOR
        }

        __syncthreads();
    }
}
}


void tiled_tc_mm_32 (
    float* A,
    float* B,
    float* C,
    size_t M, size_t N, size_t K
) {

}