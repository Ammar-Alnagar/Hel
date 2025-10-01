#include "safetensors_loader.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <nlohmann/json.hpp> // For JSON parsing

namespace safetensors {

// Safetensors format constants
const uint8_t SAFETENSORS_MAGIC[8] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

DType string_to_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32") return DType::FP32;
    if (dtype_str == "F16") return DType::FP16;
    if (dtype_str == "I8") return DType::INT8;
    if (dtype_str == "Q4") return DType::Q4;
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::FP32: return "F32";
        case DType::FP16: return "F16";
        case DType::INT8: return "I8";
        case DType::Q4: return "Q4";
        default: return "F32";
    }
}

bool is_valid_safetensors(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    uint8_t magic[8];
    file.read(reinterpret_cast<char*>(magic), 8);

    return std::memcmp(magic, SAFETENSORS_MAGIC, 8) == 0;
}

SafeTensorsHeader inspect_safetensors(const std::string& filepath) {
    if (!is_valid_safetensors(filepath)) {
        throw std::runtime_error("Invalid Safetensors file: " + filepath);
    }

    std::ifstream file(filepath, std::ios::binary);

    // Skip magic bytes
    file.seekg(8);

    // Read header length
    uint64_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 8);

    // Read header JSON
    std::string header_json(header_len, '\0');
    file.read(&header_json[0], header_len);

    // Parse JSON
    auto json_data = nlohmann::json::parse(header_json);

    SafeTensorsHeader header;

    // Parse tensor metadata
    if (json_data.contains("tensors") && json_data["tensors"].is_object()) {
        for (const auto& [name, tensor_info] : json_data["tensors"].items()) {
            if (tensor_info.contains("shape") && tensor_info["shape"].is_array()) {
                std::vector<int64_t> shape;
                for (const auto& dim : tensor_info["shape"]) {
                    shape.push_back(dim);
                }
                header.shape_map[name] = shape;
            }

            if (tensor_info.contains("dtype")) {
                header.dtype_map[name] = tensor_info["dtype"];
            }
        }
    }

    // Parse metadata
    if (json_data.contains("metadata") && json_data["metadata"].is_object()) {
        for (const auto& [key, value] : json_data["metadata"].items()) {
            header.metadata[key] = value;
        }
    }

    return header;
}

std::unordered_map<std::string, Tensor> load_safetensors(const std::string& filepath) {
    std::cout << "Loading Safetensors model: " << filepath << std::endl;

    SafeTensorsHeader header = inspect_safetensors(filepath);
    std::ifstream file(filepath, std::ios::binary);

    // Skip magic and header
    file.seekg(8);
    uint64_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    file.seekg(header_len, std::ios::cur);

    std::unordered_map<std::string, Tensor> tensors;

    for (const auto& [name, shape] : header.shape_map) {
        auto it = header.dtype_map.find(name);
        if (it == header.dtype_map.end()) {
            continue;
        }

        DType dtype = string_to_dtype(it->second);

        // Calculate tensor size
        size_t numel = 1;
        for (int64_t dim : shape) {
            numel *= dim;
        }

        Tensor tensor(std::vector<int>(shape.begin(), shape.end()), dtype);

        // Read tensor data
        size_t bytes_to_read = tensor.byte_size();
        file.read(reinterpret_cast<char*>(tensor.raw()), bytes_to_read);

        tensors[name] = std::move(tensor);

        std::cout << "Loaded tensor: " << name
                  << " shape: [" << shape[0];
        for (size_t i = 1; i < shape.size(); ++i) {
            std::cout << ", " << shape[i];
        }
        std::cout << "] dtype: " << it->second << std::endl;
    }

    std::cout << "Safetensors model loaded with " << tensors.size() << " tensors" << std::endl;
    return tensors;
}

} // namespace safetensors
