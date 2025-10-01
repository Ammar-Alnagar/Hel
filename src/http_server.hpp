#ifndef HTTP_SERVER_HPP
#define HTTP_SERVER_HPP

#include "app.hpp"
#include <string>
#include <thread>
#include <atomic>
#include <memory>

class HTTPServer {
public:
    HTTPServer(int port = 8080);
    ~HTTPServer();

    // Start the server
    void start();

    // Stop the server
    void stop();

    // Check if server is running
    bool is_running() const { return running_; }

private:
    void server_loop();
    std::string handle_request(const std::string& request);

    int port_;
    int server_socket_;
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> server_thread_;

    // Model state
    std::string current_model_path_;
    bool model_loaded_;

    // Simple JSON response helpers
    std::string json_response(const std::string& data);
    std::string json_error(const std::string& message);
};

#endif // HTTP_SERVER_HPP
