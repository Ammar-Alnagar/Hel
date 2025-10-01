#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

enum class DType {
    FP32,
    FP16,
    INT8,
    Q4
};

class Tensor {
public:
    // Constructor
    Tensor(const std::vector<int>& shape, DType dtype);

    // Destructor
    ~Tensor();

    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // Basic accessors
    void* raw() { return data_.get(); }
    size_t byte_size() const { return byte_size_; }
    std::vector<int> shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const { return numel_; }

    // Templated data access with type checking
    template<typename T>
    T* data() {
        // For Q4, we need special handling since it doesn't map to a standard type
        if (dtype_ == DType::Q4) {
            throw std::runtime_error("Q4 tensors should use raw() access for uint8_t data");
        }
        if (sizeof(T) != element_size()) {
            throw std::runtime_error("Type size mismatch");
        }
        return static_cast<T*>(data_.get());
    }

    template<typename T>
    const T* data() const {
        // For Q4, we need special handling since it doesn't map to a standard type
        if (dtype_ == DType::Q4) {
            throw std::runtime_error("Q4 tensors should use raw() access for uint8_t data");
        }
        if (sizeof(T) != element_size()) {
            throw std::runtime_error("Type size mismatch");
        }
        return static_cast<const T*>(data_.get());
    }

    // Special accessor for Q4 data (packed uint8_t)
    uint8_t* q4_data() {
        if (dtype_ != DType::Q4) {
            throw std::runtime_error("q4_data() only valid for Q4 tensors");
        }
        return static_cast<uint8_t*>(data_.get());
    }

    const uint8_t* q4_data() const {
        if (dtype_ != DType::Q4) {
            throw std::runtime_error("q4_data() only valid for Q4 tensors");
        }
        return static_cast<const uint8_t*>(data_.get());
    }

    // Shape utilities
    size_t element_size() const;
    bool is_contiguous() const { return true; } // For now, assume contiguous

    // Reshape (creates new tensor)
    Tensor reshape(const std::vector<int>& new_shape) const;

    // String representation for debugging
    std::string to_string() const;

private:
    std::vector<int> shape_;
    DType dtype_;
    size_t numel_;
    size_t byte_size_;
    std::unique_ptr<uint8_t[]> data_;

    size_t calculate_numel(const std::vector<int>& shape) const;
    size_t calculate_byte_size(size_t numel, DType dtype) const;
};

#endif // TENSOR_HPP
