#ifndef APP_HPP
#define APP_HPP

#include <string>
#include <vector>

struct InferenceArgs {
    std::string model_path;
    std::string prompt;
    int max_tokens = 16;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    int seed = -1;
    bool verbose = false;
};

class App {
public:
    static int run(const InferenceArgs& args);

private:
    static std::vector<int> generate(const InferenceArgs& args);
    static int sample_token(const std::vector<float>& logits, const InferenceArgs& args);
    static void print_usage(const char* program_name);
};

#endif // APP_HPP
