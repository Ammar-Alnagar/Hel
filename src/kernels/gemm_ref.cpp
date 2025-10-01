#include "gemm_ref.hpp"
#include <stdexcept>

#ifdef USE_OPENBLAS
extern "C" {
    #include <cblas.h>
}
#endif

// Basic matrix operations for when Eigen is not available
namespace {
    void simple_matmul(const float* A, const float* B, float* C,
                      int M, int K, int N, float alpha, float beta) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = alpha * sum + beta * C[m * N + n];
            }
        }
    }
}

void GemmRef::validate_shapes(const Tensor& A, const Tensor& B, const Tensor& C) {
    // A: [M, K], B: [K, N], C: [M, N]
    auto shape_A = A.shape();
    auto shape_B = B.shape();
    auto shape_C = C.shape();

    if (shape_A.size() != 2 || shape_B.size() != 2 || shape_C.size() != 2) {
        throw std::runtime_error("GEMM requires 2D tensors");
    }

    if (shape_A[1] != shape_B[0]) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    if (shape_A[0] != shape_C[0] || shape_B[1] != shape_C[1]) {
        throw std::runtime_error("Output matrix dimensions don't match");
    }
}

void GemmRef::matmul(const Tensor& A, const Tensor& B, Tensor& C,
                    float alpha, float beta) {
    validate_shapes(A, B, C);

    auto shape_A = A.shape();
    auto shape_B = B.shape();
    int M = shape_A[0];
    int K = shape_A[1];
    int N = shape_B[1];

#ifdef USE_EIGEN
    // Use Eigen for the actual computation
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A_map(A.data<float>(), M, K);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        B_map(B.data<float>(), K, N);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>>
        C_map(C.data<float>(), M, N);

    C_map = alpha * A_map * B_map + beta * C_map;
#else
    // Use simple fallback implementation
    simple_matmul(A.data<float>(), B.data<float>(), C.data<float>(), M, K, N, alpha, beta);
#endif
}

void GemmRef::matvec(const Tensor& A, const Tensor& x, Tensor& y,
                    float alpha, float beta) {
    // A: [M, K], x: [K], y: [M]
    auto shape_A = A.shape();
    auto shape_x = x.shape();
    auto shape_y = y.shape();

    if (shape_A.size() != 2 || shape_x.size() != 1 || shape_y.size() != 1) {
        throw std::runtime_error("matvec requires A to be 2D and x, y to be 1D");
    }

    if (shape_A[1] != shape_x[0] || shape_A[0] != shape_y[0]) {
        throw std::runtime_error("Matrix-vector dimensions don't match");
    }

    int M = shape_A[0];
    int K = shape_A[1];

#ifdef USE_EIGEN
    // Use Eigen for computation
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A_map(A.data<float>(), M, K);
    Eigen::Map<const Eigen::Vector<float>>
        x_map(x.data<float>(), K);
    Eigen::Map<Eigen::Vector<float>>
        y_map(y.data<float>(), M);

    y_map = alpha * A_map * x_map + beta * y_map;
#else
    // Use simple fallback implementation
    const float* A_data = A.data<float>();
    const float* x_data = x.data<float>();
    float* y_data = y.data<float>();

    for (int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_data[m * K + k] * x_data[k];
        }
        y_data[m] = alpha * sum + beta * y_data[m];
    }
#endif
}
