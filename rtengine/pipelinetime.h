// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace rtengine {

// Opt-in wall-clock profiling. Labels are internal constants, never filenames.
// Nested scopes are inclusive and must not be added to their children.
class PipelineTimer {
public:
    PipelineTimer(const char *name, int width = 0, int height = 0, int pipeline = -1):
        name_(name), width_(width), height_(height), pipeline_(pipeline), active_(enabled())
    {
        if (active_) {
            start_ = Clock::now();
        }
    }

    ~PipelineTimer()
    {
        if (active_) {
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_).count();
            static std::mutex output_mutex;
            std::lock_guard<std::mutex> lock(output_mutex);
            std::fprintf(stderr, "ART_PROFILE {\"name\":\"%s\",\"width\":%d,\"height\":%d,\"pipeline\":%d,\"us\":%lld}\n",
                         name_, width_, height_, pipeline_, static_cast<long long>(us));
        }
    }

    PipelineTimer(const PipelineTimer&) = delete;
    PipelineTimer& operator=(const PipelineTimer&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    static bool enabled()
    {
        static const bool value = [] {
            const char *env = std::getenv("ART_PROFILE");
            return env && std::strcmp(env, "1") == 0;
        }();
        return value;
    }
    const char *name_;
    int width_, height_, pipeline_;
    bool active_;
    Clock::time_point start_;
};

} // namespace rtengine
