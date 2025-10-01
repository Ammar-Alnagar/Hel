#ifndef SAFETENSORS_LOADER_HPP
#define SAFETENSORS_LOADER_HPP

#include "../tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace safetensors {

// Safetensors header structure
struct SafeTensorsHeader {
    std::unordered_map<std::string, std::vector<int64_t>> shape_map;
    std::unordered_map<std::string, std::string> dtype_map;
    std::unordered_map<std::string, std::vector<int64_t>> offset_map;
    std::unordered_map<std::string, std::string> metadata;
};

// Load Safetensors model
std::unordered_map<std::string, Tensor> load_safetensors(const std::string& filepath);

// Get model metadata without loading tensors
SafeTensorsHeader inspect_safetensors(const std::string& filepath);

// Convert string dtype to our DType
DType string_to_dtype(const std::string& dtype_str);

// Convert our DType to string
std::string dtype_to_string(DType dtype);

// Validate Safetensors file format
bool is_valid_safetensors(const std::string& filepath);

} // namespace safetensors

#endif // SAFETENSORS_LOADER_HPP
