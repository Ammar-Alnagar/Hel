#include "tensor.hpp"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>

Tensor::Tensor(const std::vector<int>& shape, DType dtype)
    : shape_(shape), dtype_(dtype) {
    numel_ = calculate_numel(shape);
    byte_size_ = calculate_byte_size(numel_, dtype);

    // Allocate aligned memory
    data_ = std::make_unique<uint8_t[]>(byte_size_);
}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      numel_(other.numel_),
      byte_size_(other.byte_size_),
      data_(std::move(other.data_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        byte_size_ = other.byte_size_;
        data_ = std::move(other.data_);
    }
    return *this;
}

size_t Tensor::calculate_numel(const std::vector<int>& shape) const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<int>());
}

size_t Tensor::calculate_byte_size(size_t numel, DType dtype) const {
    switch (dtype) {
        case DType::FP32:
            return numel * 4;
        case DType::FP16:
            return numel * 2;
        case DType::INT8:
            return numel * 1;
        case DType::Q4:
            // Q4 stores 2 elements per byte (4 bits each)
            return (numel + 1) / 2;
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

size_t Tensor::element_size() const {
    switch (dtype_) {
        case DType::FP32:
            return 4;
        case DType::FP16:
            return 2;
        case DType::INT8:
            return 1;
        case DType::Q4:
            return 0.5; // Special case for Q4
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_numel = calculate_numel(new_shape);
    if (new_numel != numel_) {
        throw std::runtime_error("Cannot reshape: total elements don't match");
    }

    Tensor reshaped(new_shape, dtype_);
    std::memcpy(reshaped.data_.get(), data_.get(), byte_size_);
    return reshaped;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], dtype=";

    switch (dtype_) {
        case DType::FP32: oss << "FP32"; break;
        case DType::FP16: oss << "FP16"; break;
        case DType::INT8: oss << "INT8"; break;
        case DType::Q4: oss << "Q4"; break;
    }

    oss << ", numel=" << numel_ << ")";
    return oss.str();
}
