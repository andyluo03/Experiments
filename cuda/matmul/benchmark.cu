#include <iostream>
#include <chrono>
#include <random>
#include <functional>

#include "naive/naive_gemm.hpp"

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
    }

    void benchmarkDevice (const std::string& kernel_name, std::function<void(float*, float*, float*, int, int, int)> matmul) {
      float *d_A, *d_B, *d_C;

      cudaMalloc(&d_A, sizeof(float) * M_ * N_);
      cudaMalloc(&d_B, sizeof(float) * N_ * K_);
      cudaMalloc(&d_C, sizeof(float) * M_ * K_);

      cudaMemcpy(d_A, A_, sizeof(float) * M_ * N_, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B_, sizeof(float) * N_ * K_, cudaMemcpyHostToDevice);

      auto start = std::chrono::high_resolution_clock::now();
      matmul(d_A, d_B, d_C, M_, N_, K_);
      auto end = std::chrono::high_resolution_clock::now();

      cudaMemcpy(d_C, C_, sizeof(float) * M_ * K_, cudaMemcpyDeviceToHost);

      cudaFree(&d_A);
      cudaFree(&d_B);
      cudaFree(&d_C);

      long long walltime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      double realized_flops = (static_cast<double>(M_) * N_ * K_ * 2.0) / static_cast<double>(walltime_ns);

      std::cout << "[" << name_ << " -- " << kernel_name << "]: GFLOPs: " << realized_flops << ", Runtime (ns): " << walltime_ns << std::endl;
    }

    ~Benchmark() {
      delete A_;
      delete B_;
      delete C_;
    }

  private:
    float* A_;
    float* B_;
    float* C_;
    int M_;
    int N_;
    int K_;

    std::string name_;
};


int main () {
    Benchmark f32_4096("4096, 4096, 4096", 4096, 4096, 4096);

    f32_4096.benchmarkDevice("Naive", &naive_gemm_32);
}