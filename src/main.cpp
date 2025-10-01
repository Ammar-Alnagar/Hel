#include "app.hpp"
#include <iostream>
#include <string>
#include <vector>

InferenceArgs parse_args(int argc, char* argv[]) {
    InferenceArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help") {
            App::print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            App::print_usage(argv[0]);
            exit(1);
        }
    }

    // Validate required arguments
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        App::print_usage(argv[0]);
        exit(1);
    }

    if (args.prompt.empty()) {
        std::cerr << "Error: --prompt is required" << std::endl;
        App::print_usage(argv[0]);
        exit(1);
    }

    return args;
}

int main(int argc, char* argv[]) {
    InferenceArgs args = parse_args(argc, argv);

    std::cout << "Helios Engine - Mini LLM Inference" << std::endl;
    std::cout << "===================================" << std::endl;

    return App::run(args);
}
