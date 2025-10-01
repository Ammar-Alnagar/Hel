#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <chrono>
#include <string>
#include <unordered_map>

class Profiler {
public:
    static Profiler& instance();

    void start(const std::string& name);
    void stop(const std::string& name);

    struct Timing {
        double total_time = 0.0;
        int count = 0;
        double avg_time() const { return count > 0 ? total_time / count : 0.0; }
    };

    const std::unordered_map<std::string, Timing>& get_timings() const {
        return timings_;
    }

    void reset();

    void print_summary() const;

private:
    Profiler() = default;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::unordered_map<std::string, Timing> timings_;
    mutable std::mutex mutex_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string name_;
};

#define PROFILE_SCOPE(name) ScopedTimer timer_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#endif // PROFILER_HPP
