#ifndef GGUF_LOADER_HPP
#define GGUF_LOADER_HPP

#include "../tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace gguf {

// GGUF tensor info
struct GGUFMetadata {
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<std::string> tensor_names;
    std::unordered_map<std::string, std::vector<int>> tensor_shapes;
    std::unordered_map<std::string, std::string> tensor_types;
};

// GGUF file format constants
const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
const uint32_t GGUF_VERSION = 3;

// GGML tensor types
enum GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    COUNT = 19
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dimensions;
    GGMLType type;
    uint64_t offset;
    uint64_t size_bytes;
};

// Load GGUF model and return tensor map
std::unordered_map<std::string, Tensor> load_gguf_model(const std::string& filepath);

// Get model metadata without loading tensors
GGUFMetadata inspect_gguf_model(const std::string& filepath);

// Convert GGML type to our DType
DType ggml_to_dtype(GGMLType ggml_type);

// Get size of GGML type in bytes per element
size_t ggml_type_size(GGMLType type);

} // namespace gguf

#endif // GGUF_LOADER_HPP
