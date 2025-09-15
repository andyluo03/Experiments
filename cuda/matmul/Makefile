NVCC := nvcc

SRCS := benchmark.cu naive/naive_gemm.cu
OUT := bench.out

all:
	$(NVCC) $(SRCS) -o $(OUT)