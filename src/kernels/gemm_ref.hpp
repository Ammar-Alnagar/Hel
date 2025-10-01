#ifndef GEMM_REF_HPP
#define GEMM_REF_HPP

#include "../tensor.hpp"
#include <vector>

class GemmRef {
public:
    // Matrix multiplication: C = A * B + beta * C
    static void matmul(const Tensor& A, const Tensor& B, Tensor& C,
                      float alpha = 1.0f, float beta = 0.0f);

    // Matrix-vector multiplication: y = A * x + beta * y
    static void matvec(const Tensor& A, const Tensor& x, Tensor& y,
                      float alpha = 1.0f, float beta = 0.0f);

private:
    static void validate_shapes(const Tensor& A, const Tensor& B, const Tensor& C);
};

#endif // GEMM_REF_HPP
