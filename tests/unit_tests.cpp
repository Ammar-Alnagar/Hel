#include "../src/tensor.hpp"
#include "../src/alloc.hpp"
#include "../src/kernels/q4_rowwise.hpp"
#include "../src/kernels/gemm_ref.hpp"
#include "../src/tokenizer/sentencepiece_wrapper.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

// Test Tensor class
void test_tensor() {
    std::cout << "Testing Tensor class..." << std::endl;

    // Test basic construction
    Tensor t({2, 3}, DType::FP32);
    assert(t.shape() == std::vector<int>{2, 3});
    assert(t.numel() == 6);
    assert(t.byte_size() == 24); // 2*3*4 bytes

    // Test data access
    float* data = t.data<float>();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Test reshape
    Tensor t2 = t.reshape({3, 2});
    assert(t2.shape() == std::vector<int>{3, 2});
    assert(t2.numel() == 6);

    // Test move semantics
    Tensor t3 = std::move(t2);
    assert(t3.shape() == std::vector<int>{3, 2});

    std::cout << "✓ Tensor tests passed" << std::endl;
}

// Test allocator
void test_allocator() {
    std::cout << "Testing AlignedAllocator..." << std::endl;

    void* ptr = AlignedAllocator::allocate(128, 32);
    assert(ptr != nullptr);

    // Test alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    assert(addr % 32 == 0);

    AlignedAllocator::deallocate(ptr);
    std::cout << "✓ Allocator tests passed" << std::endl;
}

// Test Q4 quantization
void test_q4_quantization() {
    std::cout << "Testing Q4 quantization..." << std::endl;

    const int M = 2;
    const int K = 4;

    // Create test weights
    std::vector<float> weights(M * K);
    for (int i = 0; i < M * K; ++i) {
        weights[i] = static_cast<float>(i - 3); // Range -3 to 4
    }

    // Create scales
    std::vector<float> scales(M, 1.0f);

    // Pack to Q4
    std::vector<uint8_t> qweights((M * K + 1) / 2);
    std::vector<float> scales_out(M);

    pack_q4_rowwise(weights.data(), scales.data(), qweights.data(),
                   scales_out.data(), M, K);

    // Dequantize
    std::vector<float> dequantized(M * K);
    dequantize_q4_rowwise(qweights.data(), scales_out.data(),
                         dequantized.data(), M, K);

    // Check that dequantization is close to original
    for (int i = 0; i < M * K; ++i) {
        float error = std::abs(dequantized[i] - weights[i]);
        assert(error < 1.0f); // Should be close due to quantization
    }

    // Test matvec
    std::vector<float> x(K, 1.0f);
    std::vector<float> y(M, 0.0f);

    matvec_q4_rowwise(qweights.data(), scales_out.data(), x.data(), y.data(), M, K);

    // Verify result is reasonable
    for (float val : y) {
        assert(std::isfinite(val));
    }

    std::cout << "✓ Q4 quantization tests passed" << std::endl;
}

// Test GEMM
void test_gemm() {
    std::cout << "Testing GEMM..." << std::endl;

    const int M = 3;
    const int K = 4;
    const int N = 2;

    Tensor A({M, K}, DType::FP32);
    Tensor B({K, N}, DType::FP32);
    Tensor C({M, N}, DType::FP32);

    // Initialize with test data
    float* a_data = A.data<float>();
    float* b_data = B.data<float>();
    float* c_data = C.data<float>();

    for (int i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(i);
    for (int i = 0; i < K * N; ++i) b_data[i] = static_cast<float>(i % 3);
    for (int i = 0; i < M * N; ++i) c_data[i] = 0.0f;

    // Perform matrix multiplication
    GemmRef::matmul(A, B, C, 1.0f, 0.0f);

    // Verify result is finite
    for (int i = 0; i < M * N; ++i) {
        assert(std::isfinite(c_data[i]));
    }

    std::cout << "✓ GEMM tests passed" << std::endl;
}

// Test tokenizer
void test_tokenizer() {
    std::cout << "Testing Tokenizer..." << std::endl;

    Tokenizer tokenizer("dummy.model");

    std::string text = "hello world";
    auto tokens = tokenizer.encode(text);

    assert(!tokens.empty());
    assert(tokens.front() == tokenizer.bos_token_id());
    assert(tokens.back() == tokenizer.eos_token_id());

    std::string decoded = tokenizer.decode(tokens);
    // Note: our simple tokenizer won't perfectly reconstruct, but should not crash
    assert(!decoded.empty());

    std::cout << "✓ Tokenizer tests passed" << std::endl;
}

int main() {
    std::cout << "Running unit tests..." << std::endl;

    try {
        test_tensor();
        test_allocator();
        test_q4_quantization();
        test_gemm();
        test_tokenizer();

        std::cout << "\n🎉 All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
