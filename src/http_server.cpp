#include "http_server.hpp"
#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <json.hpp> // Assuming nlohmann/json or similar

// For now, we'll use a simple JSON-like response format
namespace {
    std::string create_json_response(const std::string& status, const std::string& message = "") {
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n";
        oss << "Content-Type: application/json\r\n";
        oss << "Access-Control-Allow-Origin: *\r\n";
        oss << "\r\n";
        oss << "{\n";
        oss << "  \"status\": \"" << status << "\"";
        if (!message.empty()) {
            oss << ",\n  \"message\": \"" << message << "\"";
        }
        oss << "\n}\n";
        return oss.str();
    }

    std::string create_completion_response(const std::string& text) {
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n";
        oss << "Content-Type: application/json\r\n";
        oss << "Access-Control-Allow-Origin: *\r\n";
        oss << "\r\n";
        oss << "{\n";
        oss << "  \"text\": \"" << text << "\"\n";
        oss << "}\n";
        return oss.str();
    }
}

HTTPServer::HTTPServer(int port) : port_(port), server_socket_(-1), running_(false), model_loaded_(false) {}

HTTPServer::~HTTPServer() {
    stop();
}

void HTTPServer::start() {
    // Create socket
    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    // Bind to port
    sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);

    if (bind(server_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(server_socket_);
        throw std::runtime_error("Failed to bind to port " + std::to_string(port_));
    }

    // Listen for connections
    if (listen(server_socket_, 5) < 0) {
        close(server_socket_);
        throw std::runtime_error("Failed to listen on socket");
    }

    running_ = true;
    server_thread_ = std::make_unique<std::thread>(&HTTPServer::server_loop, this);

    std::cout << "🚀 HTTP server started on port " << port_ << std::endl;
}

void HTTPServer::stop() {
    running_ = false;
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    if (server_socket_ >= 0) {
        close(server_socket_);
        server_socket_ = -1;
    }
}

void HTTPServer::server_loop() {
    std::cout << "HTTP server listening for connections..." << std::endl;

    while (running_) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_socket = accept(server_socket_, (sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            if (running_) {
                std::cerr << "Failed to accept connection" << std::endl;
            }
            continue;
        }

        // Handle client request in a simple way
        char buffer[4096];
        memset(buffer, 0, sizeof(buffer));

        ssize_t bytes_read = read(client_socket, buffer, sizeof(buffer) - 1);
        if (bytes_read > 0) {
            std::string request(buffer, bytes_read);
            std::string response = handle_request(request);

            write(client_socket, response.c_str(), response.length());
        }

        close(client_socket);
    }
}

std::string HTTPServer::handle_request(const std::string& request) {
    std::istringstream iss(request);
    std::string method, path, http_version;

    iss >> method >> path >> http_version;

    // Simple routing
    if (path == "/health") {
        return create_json_response("healthy");
    }
    else if (path == "/load") {
        // Extract model path from query parameters (simplified)
        size_t model_pos = request.find("model=");
        if (model_pos != std::string::npos) {
            size_t start = model_pos + 6;
            size_t end = request.find("&", start);
            if (end == std::string::npos) end = request.find(" ", start);
            std::string model_path = request.substr(start, end - start);

            try {
                // Load model (simplified - would need proper implementation)
                model_loaded_ = true;
                current_model_path_ = model_path;
                return create_json_response("loaded", "Model loaded successfully");
            } catch (const std::exception& e) {
                return create_json_response("error", e.what());
            }
        }
    }
    else if (path == "/generate" && method == "POST") {
        if (!model_loaded_) {
            return create_json_response("error", "No model loaded");
        }

        // Extract prompt from JSON body (simplified parsing)
        size_t body_start = request.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            std::string body = request.substr(body_start + 4);

            // Simple JSON parsing (would use proper JSON library in production)
            size_t prompt_start = body.find("\"prompt\":\"");
            if (prompt_start != std::string::npos) {
                prompt_start += 10;
                size_t prompt_end = body.find("\"", prompt_start);
                std::string prompt = body.substr(prompt_start, prompt_end - prompt_start);

                try {
                    // Generate text (simplified - would use actual inference)
                    std::string generated_text = "Generated response for: " + prompt;

                    return create_completion_response(generated_text);
                } catch (const std::exception& e) {
                    return create_json_response("error", e.what());
                }
            }
        }
    }

    // Default response
    std::ostringstream oss;
    oss << "HTTP/1.1 404 Not Found\r\n";
    oss << "Content-Type: application/json\r\n";
    oss << "\r\n";
    oss << "{\n";
    oss << "  \"error\": \"Endpoint not found\"\n";
    oss << "}\n";
    return oss.str();
}
