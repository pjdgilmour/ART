// SPDX-License-Identifier: GPL-3.0-or-later
// Compare with the actual CPU implementation, compiled without ART_USE_CUDA.
#include "../../rtengine/deconvolution.h"
#ifdef ART_TEST_OPENCL
#include "../opencl-tests/backend.h"
namespace testgpu = rtengine::opencl;
#else
#include "../../rtengine/cuda/deconvolution.h"
namespace testgpu = rtengine::cuda;
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> struct Plane {
    int w, h, stride;
    std::vector<T> data;
    std::vector<T*> rows;
    Plane(int width, int height): w(width), h(height), stride(width + 7), data(size_t(stride) * h), rows(h) {
        for (int y = 0; y < h; ++y) rows[y] = data.data() + size_t(y) * stride;
    }
};

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv)
{
    try {
        const bool no_device = argc > 1 && std::string(argv[1]) == "--no-device";
        const bool recursive = argc > 1 && std::string(argv[1]) == "--recursive";
        int w = 1025, h = 1027;
        if (argc == 3) { w = std::stoi(argv[1]); h = std::stoi(argv[2]); }
        if (recursive && argc == 4) { w = std::stoi(argv[2]); h = std::stoi(argv[3]); }
        Plane<float> cpu(w, h), gpu(w, h), blend(w, h);
        Plane<char> impulse(w, h);
        const std::vector<double> sigmas = recursive ? std::vector<double>{1.150001, 1.27, 1.31, 2.0, 2.5, 4.0} : std::vector<double>{0.25, 0.4, 0.59999, 0.6, 0.84, 0.84001, 0.93307381868362427, 1.15};
        for (size_t test = 0; test < sigmas.size(); ++test) {
            unsigned int random = 42;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    random = random * 1664525u + 1013904223u;
                    const float noise = float(random >> 16) / 65535.f;
                    float v = (x < w/2 ? 100.f : 60000.f) + noise * 100.f + y * 0.3f;
                    if (test == 0) v = 2000.f; // flat field
                    if (test == 1) v = ((x + y) % 13 == 0) ? 100000.f : -500.f;
                    if (test == 3) v = noise * 100000.f;
                    cpu.rows[y][x] = gpu.rows[y][x] = v;
                    blend.rows[y][x] = float((x + y) % 11) / 10.f;
                    impulse.rows[y][x] = ((x + y) % 17 == 0) ? 1 : 0;
                }
            }
            const auto original = gpu.data;
            auto start = Clock::now();
            const bool used = testgpu::deconvolve(gpu.rows.data(), blend.rows.data(), impulse.rows.data(), w, h, sigmas[test], 0.7f);
            const double gpu_ms = ms(start);
            if (no_device) {
                require(!used && gpu.data == original, "missing GPU must leave input untouched");
                std::puts("PASS no-device fallback, input untouched");
                return 0;
            }
            require(used, "GPU backend not exercised (device unavailable, disabled or dimensions outside supported range)");
            start = Clock::now();
            rtengine::deconvsharpening(cpu.rows.data(), blend.rows.data(), impulse.rows.data(), w, h, sigmas[test], 0.7f, true);
            const double cpu_ms = ms(start);
            double maximum = 0, total = 0;
            size_t different = 0;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const double a = cpu.rows[y][x], b = gpu.rows[y][x];
                    require(std::isfinite(a) && std::isfinite(b), "non-finite result");
                    const double error = std::abs(a - b);
                    maximum = std::max(maximum, error); total += error; different += a != b;
                    require(error <= 0.05 + 0.000005 * std::abs(a), "CPU/GPU numerical mismatch");
                }
                for (int x = w; x < gpu.stride; ++x) require(gpu.rows[y][x] == 0, "row padding overwritten");
            }
            std::printf("PASS %dx%d sigma=%.8f cpu=%.3fms gpu_total=%.3fms max_abs=%.9g mean_abs=%.9g different=%zu\n",
                        w, h, sigmas[test], cpu_ms, gpu_ms, maximum, total / (double(w)*h), different);
            std::fflush(stdout);
        }
        const auto original = gpu.data;
        for (double sigma : {0.2, recursive ? 4.000001 : 1.150001, std::numeric_limits<double>::quiet_NaN()}) {
            require(!testgpu::deconvolve(gpu.rows.data(), blend.rows.data(), impulse.rows.data(), w, h, sigma, 1.f), "unsupported sigma accepted");
            require(gpu.data == original, "fallback changed input");
        }
        require(!testgpu::deconvolve(gpu.rows.data(), blend.rows.data(), impulse.rows.data(), 16, 16, 0.8, 1.f), "small preview accepted");
        require(gpu.data == original, "small-preview fallback changed input");
        gpu.rows[0][0] = std::numeric_limits<float>::infinity();
        const auto nonfinite = gpu.data;
        require(!testgpu::deconvolve(gpu.rows.data(), blend.rows.data(), impulse.rows.data(), w, h, 0.8, 1.f), "non-finite input accepted");
        require(gpu.data == nonfinite, "non-finite fallback changed input");
        std::puts("PASS unsupported/small/non-finite fallback");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
