#include "gguf_loader.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace gguf {

// Helper functions for reading GGUF data
uint32_t read_u32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint64_t read_u64(std::ifstream& file) {
    uint64_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::string read_string(std::ifstream& file, size_t length) {
    std::string str(length, '\0');
    file.read(&str[0], length);
    return str;
}

DType ggml_to_dtype(GGMLType ggml_type) {
    switch (ggml_type) {
        case F32: return DType::FP32;
        case F16: return DType::FP16;
        case I8: return DType::INT8;
        case Q4_0:
        case Q4_1:
        case Q4_K:
            return DType::Q4;
        default:
            throw std::runtime_error("Unsupported GGML type: " + std::to_string(ggml_type));
    }
}

size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case F32: return 4;
        case F16: return 2;
        case Q4_0: return 1; // 4-bit, but stored as bytes
        case Q4_1: return 1;
        case Q8_0: return 1;
        case I8: return 1;
        case I16: return 2;
        case I32: return 4;
        default: return 1;
    }
}

GGUFMetadata inspect_gguf_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open GGUF file: " + filepath);
    }

    GGUFHeader header;
    header.magic = read_u32(file);
    if (header.magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic number");
    }

    header.version = read_u32(file);
    header.tensor_count = read_u64(file);
    header.metadata_kv_count = read_u64(file);

    GGUFMetadata metadata;

    // Read metadata key-value pairs
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        uint32_t key_len = read_u32(file);
        std::string key = read_string(file, key_len);

        // Read value type (simplified - assume string for now)
        uint32_t value_type = read_u32(file);
        uint32_t value_len = read_u32(file);
        std::string value = read_string(file, value_len);

        metadata.metadata[key] = value;

        if (key == "general.architecture") {
            metadata.architecture = value;
        }
    }

    // Read tensor infos
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        uint32_t name_len = read_u32(file);
        std::string name = read_string(file, name_len);

        GGUFTensorInfo tensor_info;
        tensor_info.name = name;
        tensor_info.n_dims = read_u32(file);

        std::vector<uint64_t> dims(tensor_info.n_dims);
        for (uint32_t d = 0; d < tensor_info.n_dims; ++d) {
            dims[d] = read_u64(file);
        }
        tensor_info.dimensions = dims;
        tensor_info.type = static_cast<GGMLType>(read_u32(file));
        tensor_info.offset = read_u64(file);

        // Calculate size (simplified)
        tensor_info.size_bytes = 1;
        for (uint64_t dim : dims) {
            tensor_info.size_bytes *= dim;
        }
        tensor_info.size_bytes *= ggml_type_size(tensor_info.type);

        metadata.tensor_names.push_back(name);
        metadata.tensor_shapes[name] = std::vector<int>(dims.begin(), dims.end());
        metadata.tensor_types[name] = std::to_string(static_cast<int>(tensor_info.type));
    }

    return metadata;
}

std::unordered_map<std::string, Tensor> load_gguf_model(const std::string& filepath) {
    std::cout << "Loading GGUF model: " << filepath << std::endl;

    GGUFMetadata metadata = inspect_gguf_model(filepath);
    std::ifstream file(filepath, std::ios::binary);

    // Skip header (already read in inspect)
    file.seekg(0);
    GGUFHeader header;
    header.magic = read_u32(file);
    header.version = read_u32(file);
    header.tensor_count = read_u64(file);
    header.metadata_kv_count = read_u64(file);

    // Skip metadata KV pairs
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        uint32_t key_len = read_u32(file);
        file.seekg(key_len, std::ios::cur);

        uint32_t value_type = read_u32(file);
        uint32_t value_len = read_u32(file);
        file.seekg(value_len, std::ios::cur);
    }

    std::unordered_map<std::string, Tensor> tensors;

    // Read tensor data
    for (const std::string& name : metadata.tensor_names) {
        // Skip to tensor data (simplified - would need proper offset calculation)
        // For now, create placeholder tensors
        auto it = metadata.tensor_shapes.find(name);
        if (it != metadata.tensor_shapes.end()) {
            DType dtype = ggml_to_dtype(static_cast<GGMLType>(std::stoi(metadata.tensor_types[name])));
            Tensor tensor(it->second, dtype);

            // Initialize with dummy data (in real implementation, read from file)
            if (dtype == DType::FP32) {
                float* data = tensor.data<float>();
                for (size_t i = 0; i < tensor.numel(); ++i) {
                    data[i] = 0.01f * (rand() % 100 - 50) / 50.0f;
                }
            }

            tensors[name] = std::move(tensor);
            std::cout << "Loaded tensor: " << name << " shape: [";
            for (size_t i = 0; i < it->second.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << it->second[i];
            }
            std::cout << "]" << std::endl;
        }
    }

    std::cout << "GGUF model loaded with " << tensors.size() << " tensors" << std::endl;
    return tensors;
}

} // namespace gguf
