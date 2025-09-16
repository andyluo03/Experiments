#include <iostream>
#include <chrono>
#include <random>
#include <functional>

#include "naive/naive_gemm.hpp"
#include "tiled/tiled_gemm.hpp"

constexpr float kEpsilon = 0.0001f;

class Benchmark {
  public:
    Benchmark(const std::string name, int M, int N, int K) : name_{name}, M_{M}, N_{N}, K_{K} {
        A_ = new float[M * N];
        B_ = new float[N * K];
        C_ = new float[M * K];

        std::random_device rd;
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> dist(0.0f, 1e19f);

        for (int i = 0; i < M * N; i++) {
            A_[i] = dist(gen);
        }

        for (int i = 0; i < N * K; i++) {
            B_[i] = dist(gen);
        }

        cudaMalloc(&d_A_, sizeof(float) * M_ * N_);
        cudaMalloc(&d_B_, sizeof(float) * N_ * K_);
        cudaMalloc(&d_C_, sizeof(float) * M_ * K_);

        cudaMemcpy(d_A_, A_, sizeof(float) * M_ * N_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_, B_, sizeof(float) * N_ * K_, cudaMemcpyHostToDevice);
    }

    void benchmarkDevice (const std::string& kernel_name, std::function<void(float*, float*, float*, int, int, int)> matmul) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul(d_A_, d_B_, d_C_, M_, N_, K_);
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t walltime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double realized_flops = (static_cast<double>(M_) * N_ * K_ * 2.0) / static_cast<double>(walltime_ns);

        std::cout << "[" << name_ << " -- " << kernel_name << "]: GFLOPs: " << realized_flops << ", Runtime (ns): " << walltime_ns << std::endl;
    }

    void compareOracle(
      const std::string& name, 
      std::function<void(float*, float*, float*, int, int, int)> mm,
      std::function<void(float*, float*, float*, int, int, int)> oracle
    ) {
      float* C_tmp = new float[M_ * K_];

      mm(d_A_, d_B_, d_C_, M_, N_, K_);
      cudaMemcpy(d_C_, C_tmp, sizeof(float) * M_ * K_, cudaMemcpyDeviceToHost);

      oracle(d_A_, d_B_, d_C_, M_, N_, K_);
      cudaMemcpy(d_C_, C_, sizeof(float) * M_ * K_, cudaMemcpyDeviceToHost);

      for (int i = 0; i < M_ * N_; i++) {
        if (abs(C_tmp[i] - C_[i]) > kEpsilon) {
          std::cout << "[" << name << "]: Failed." << std::endl;
          delete C_tmp;
          return;
        }
      }

      std::cout << "[" << name << "]: Passed." << std::endl;
      delete C_tmp;
    }

    ~Benchmark() {
        delete A_;
        delete B_;
        delete C_;

        cudaFree(d_A_);
        cudaFree(d_B_);
        cudaFree(d_C_);
    }

  private:
    float* A_;
    float* B_;
    float* C_;

    float* d_A_;
    float* d_B_;
    float* d_C_;

    int M_;
    int N_;
    int K_;

    std::string name_;
};

int main () {
    Benchmark f32_4096("4096, 4096, 4096", 4096, 4096, 4096);

    f32_4096.benchmarkDevice("Naive", &naive_gemm_32);
    f32_4096.benchmarkDevice("Tiled", &tiled_gemm_32);

    f32_4096.compareOracle("Tiled", &tiled_gemm_32, &naive_gemm_32);
}