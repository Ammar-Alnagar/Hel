#ifndef BATCH_PROCESSOR_HPP
#define BATCH_PROCESSOR_HPP

#include "tensor.hpp"
#include "transformer/transformer.hpp"
#include <vector>
#include <string>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

struct BatchRequest {
    std::vector<int> input_tokens;
    std::string prompt;
    int max_tokens;
    float temperature;
    int top_k;
    int top_p;
    int seed;
    std::promise<std::vector<int>> result_promise;
};

struct BatchResult {
    std::vector<int> generated_tokens;
    float inference_time_ms;
    size_t memory_used_bytes;
};

class BatchProcessor {
public:
    BatchProcessor(size_t max_batch_size = 8, size_t queue_size = 100);
    ~BatchProcessor();

    // Submit a batch request for processing
    std::future<std::vector<int>> submit_request(const BatchRequest& request);

    // Start processing requests
    void start();

    // Stop processing and wait for completion
    void stop();

    // Get current queue size
    size_t queue_size() const;

private:
    void processing_loop();
    std::vector<BatchResult> process_batch(const std::vector<BatchRequest>& requests);

    size_t max_batch_size_;
    size_t queue_size_;
    std::queue<BatchRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_;

    std::unique_ptr<std::thread> processing_thread_;

    // Model state (would be loaded from file)
    std::unique_ptr<Transformer> transformer_;
};

#endif // BATCH_PROCESSOR_HPP
