NVCC := nvcc

SRCS := benchmark.cu naive/naive_mm.cu tiled/tiled_mm.cu
OUT := bench.out

all:
	$(NVCC) $(SRCS) -o $(OUT)