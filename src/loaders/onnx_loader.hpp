#ifndef ONNX_LOADER_HPP
#define ONNX_LOADER_HPP

#include "../tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, Tensor> load_onnx_initializers(const std::string& onnx_path);

struct ModelInfo {
    std::vector<std::string> initializer_names;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::unordered_map<std::string, std::vector<int>> shapes;
    std::unordered_map<std::string, DType> dtypes;
};

ModelInfo inspect_onnx_model(const std::string& onnx_path);

#endif // ONNX_LOADER_HPP
