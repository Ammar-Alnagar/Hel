#include "profiler.hpp"
#include <iostream>
#include <iomanip>
#include <mutex>

Profiler& Profiler::instance() {
    static Profiler instance;
    return instance;
}

void Profiler::start(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_times_[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::stop(const std::string& name) {
    auto end_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = start_times_.find(name);
    if (it != start_times_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second
        ).count() / 1e6; // Convert to seconds

        timings_[name].total_time += duration;
        timings_[name].count++;
        start_times_.erase(it);
    }
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
    start_times_.clear();
}

void Profiler::print_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "\n=== Profiling Summary ===\n";
    std::cout << std::setw(30) << std::left << "Operation"
              << std::setw(12) << "Count"
              << std::setw(12) << "Total(s)"
              << std::setw(12) << "Avg(s)"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    for (const auto& [name, timing] : timings_) {
        std::cout << std::setw(30) << std::left << name
                  << std::setw(12) << timing.count
                  << std::setw(12) << std::fixed << std::setprecision(6) << timing.total_time
                  << std::setw(12) << std::fixed << std::setprecision(6) << timing.avg_time()
                  << std::endl;
    }
    std::cout << std::endl;
}

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {
    Profiler::instance().start(name_);
}

ScopedTimer::~ScopedTimer() {
    Profiler::instance().stop(name_);
}
