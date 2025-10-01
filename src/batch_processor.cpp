#include "batch_processor.hpp"
#include "loaders/onnx_loader.hpp"
#include "util/profiler.hpp"
#include <algorithm>
#include <chrono>
#include <numeric>

BatchProcessor::BatchProcessor(size_t max_batch_size, size_t queue_size)
    : max_batch_size_(max_batch_size), queue_size_(queue_size), running_(false) {

    // Initialize transformer (simplified - would load from model file)
    auto weights = load_onnx_initializers("tests/golden_baselines/mini.onnx");
    if (weights.empty()) {
        std::cerr << "Warning: Using dummy model for batch processing" << std::endl;
    }
    ModelWeights model_weights{weights};
    transformer_ = std::make_unique<Transformer>(model_weights);
}

BatchProcessor::~BatchProcessor() {
    stop();
}

std::future<std::vector<int>> BatchProcessor::submit_request(const BatchRequest& request) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (request_queue_.size() >= queue_size_) {
        throw std::runtime_error("Request queue is full");
    }

    request_queue_.push(request);
    queue_cv_.notify_one();

    return std::move(request.result_promise.get_future());
}

void BatchProcessor::start() {
    if (running_) {
        return;
    }

    running_ = true;
    processing_thread_ = std::make_unique<std::thread>(&BatchProcessor::processing_loop, this);
}

void BatchProcessor::stop() {
    running_ = false;
    queue_cv_.notify_all();

    if (processing_thread_ && processing_thread_->joinable()) {
        processing_thread_->join();
    }
}

size_t BatchProcessor::queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return request_queue_.size();
}

void BatchProcessor::processing_loop() {
    std::cout << "Batch processor started" << std::endl;

    while (running_) {
        std::vector<BatchRequest> batch;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait for requests or until stopped
            queue_cv_.wait(lock, [this]() {
                return !running_ || !request_queue_.empty();
            });

            if (!running_) {
                break;
            }

            // Collect up to max_batch_size requests
            while (!request_queue_.empty() && batch.size() < max_batch_size_) {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop();
            }
        }

        if (!batch.empty()) {
            try {
                auto results = process_batch(batch);

                // Fulfill promises
                for (size_t i = 0; i < batch.size() && i < results.size(); ++i) {
                    batch[i].result_promise.set_value(results[i].generated_tokens);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing batch: " << e.what() << std::endl;

                // Set exception for all requests in batch
                for (auto& request : batch) {
                    try {
                        request.result_promise.set_exception(std::make_exception_ptr(e));
                    } catch (...) {
                        // Promise might already be set
                    }
                }
            }
        }
    }

    std::cout << "Batch processor stopped" << std::endl;
}

std::vector<BatchResult> BatchProcessor::process_batch(const std::vector<BatchRequest>& requests) {
    PROFILE_SCOPE("batch_process");

    std::vector<BatchResult> results;
    results.reserve(requests.size());

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& request : requests) {
        PROFILE_SCOPE("single_inference");

        try {
            // Simulate inference (would use actual transformer forward pass)
            std::vector<int> generated_tokens = request.input_tokens;

            // Add some dummy generated tokens
            for (int i = 0; i < request.max_tokens; ++i) {
                generated_tokens.push_back(1000 + i); // Dummy token IDs
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            float inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0f;

            results.push_back({
                generated_tokens,
                inference_time,
                1024 * 1024 // 1MB dummy memory usage
            });

        } catch (const std::exception& e) {
            std::cerr << "Error in single inference: " << e.what() << std::endl;
            results.push_back({
                request.input_tokens, // Return original tokens on error
                0.0f,
                0
            });
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0f;

    std::cout << "Processed batch of " << requests.size() << " requests in "
              << total_time << "ms" << std::endl;

    return results;
}
