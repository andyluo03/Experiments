# Matmul

Fast matrix multiplication on RTX 5070 (sm120).

I chose not to use a benchmarking library as improvements should be large.

## To-do Optimizations

- [x] Tiled Matrix Multiplication
- [ ] Register Tiling
- [ ] cp.async
- [ ] Parameter Sweeping
- [ ] Tensor Cores
- [ ] Other (CuTe, TMA, etc.)
