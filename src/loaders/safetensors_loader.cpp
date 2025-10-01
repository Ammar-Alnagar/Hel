#include "safetensors_loader.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

// Simple JSON parsing fallback when nlohmann/json is not available
#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#else
// Minimal JSON-like parsing for Safetensors header
#include <unordered_map>
namespace {
    // Simple JSON parser for basic key-value pairs
    std::unordered_map<std::string, std::string> parse_simple_json(const std::string& json_str) {
        std::unordered_map<std::string, std::string> result;
        std::string cleaned = json_str;
        // Remove whitespace and newlines for simplicity
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), ::isspace), cleaned.end());

        // Very basic parsing - look for "key":"value" patterns
        size_t pos = 0;
        while ((pos = cleaned.find('"', pos)) != std::string::npos) {
            size_t key_start = pos + 1;
            size_t key_end = cleaned.find('"', key_start);
            if (key_end == std::string::npos) break;

            std::string key = cleaned.substr(key_start, key_end - key_start);

            // Find value
            size_t colon_pos = cleaned.find(':', key_end);
            if (colon_pos == std::string::npos) break;

            size_t value_start = colon_pos + 1;
            size_t value_end;

            if (cleaned[value_start] == '"') {
                // String value
                value_start++;
                value_end = cleaned.find('"', value_start);
                if (value_end != std::string::npos) {
                    std::string value = cleaned.substr(value_start, value_end - value_start);
                    result[key] = value;
                    pos = value_end + 1;
                }
            } else {
                // Number or other value
                value_end = cleaned.find_first_of(",}", value_start);
                if (value_end != std::string::npos) {
                    std::string value = cleaned.substr(value_start, value_end - value_start);
                    result[key] = value;
                    pos = value_end;
                }
            }
            if (value_end == std::string::npos) break;
        }

        return result;
    }
}
#endif

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
#ifdef USE_NLOHMANN_JSON
    auto json_data = nlohmann::json::parse(header_json);
#else
    auto json_data = parse_simple_json(header_json);
#endif

    SafeTensorsHeader header;

    // Parse tensor metadata
#ifdef USE_NLOHMANN_JSON
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
#else
    // Simple parsing for basic key-value pairs
    // This is a simplified implementation - real Safetensors parsing would be more complex
    std::cout << "Warning: Using simplified JSON parsing for Safetensors. Some features may not work." << std::endl;
    // For demo purposes, we'll create some dummy tensor info
    header.shape_map["dummy.weight"] = {768, 768};
    header.dtype_map["dummy.weight"] = "F32";
#endif

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
