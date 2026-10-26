// SPDX-License-Identifier: GPL-3.0-or-later
// Extracted unchanged in arithmetic from ART/RawTherapee ipsharpen.cc.
// Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
#include "deconvolution.h"
#include "gauss.h"
#include "jaggedarray.h"
#include "rt_math.h"
#include "opthelper.h"
#include "StopWatch.h"
#include "pipelinetime.h"
#include "performance.h"
#ifdef ART_USE_CUDA
#include "cuda/deconvolution.h"
#include <vector>
#include <future>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace rtengine {
void deconvsharpening(float **luminance, float **blend, char **impulse, int W, int H, double sigma, float amount, bool multiThread, bool forceCPU)
{
    PipelineTimer timer("deconvsharpening", W, H);
    if (amount <= 0) {
        return;
    }
BENCHFUN

    const int maxiter = 20;
    const float delta_factor = 0.2f;

    if (sigma < 0.2f) {
        return;
    }

#ifdef ART_USE_CUDA
    const char *verify_env = std::getenv("ART_CUDA_VERIFY");
    const bool verify = verify_env && std::strcmp(verify_env, "1") == 0;
    std::vector<float> original, gpu_result;
    if (verify) {
        original.resize(size_t(W) * H);
        for (int y = 0; y < H; ++y) {
            std::memcpy(original.data() + size_t(y) * W, luminance[y], W * sizeof(float));
        }
    }
    // Measured crossover on the test workstation: the recursive GPU path loses
    // on small previews. Keep the backend callable there for explicit tests.
    const bool largeEnough = sigma <= 1.15 || size_t(W) * H >= 8 * 1024 * 1024;
    if (!forceCPU && largeEnough && cuda::deconvolve(luminance, blend, impulse, W, H, sigma, amount, multiThread)) {
        if (!verify) return;
        gpu_result.resize(size_t(W) * H);
        for (int y = 0; y < H; ++y) {
            std::memcpy(gpu_result.data() + size_t(y) * W, luminance[y], W * sizeof(float));
            std::memcpy(luminance[y], original.data() + size_t(y) * W, W * sizeof(float));
        }
        // Diagnostic mode runs the CPU oracle on exactly the same input and
        // returns that CPU result; do not use it for performance measurements.
    }
#endif

    JaggedArray<float> tmp(W, H);
    JaggedArray<float> tmpI(W, H);
    JaggedArray<float> out(W, H);

    constexpr float offset = 1000.f;

#ifdef _OPENMP
#   pragma omp parallel for if (multiThread)
#endif
    for (int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            luminance[i][j] += offset;
            tmpI[i][j] = std::max(luminance[i][j], 0.f);
            assert(std::isfinite(tmpI[i][j]));
            out[i][j] = RT_NAN;
        }
    }

    const auto get_output =
        [&](int i, int j) -> float
        {
            if (UNLIKELY(std::isnan(tmpI[i][j]))) {
                return luminance[i][j];
            }
            float b = impulse[i][j] ? 0.f : blend[i][j] * amount;
            return intp(b, std::max(tmpI[i][j], 0.0f), luminance[i][j]);
        };

    const auto check_stop =
        [&](int y, int x) -> void
        {
            if (LIKELY(std::isnan(out[y][x]))) {
                float l = luminance[y][x];
                float delta = l * delta_factor;
                if (UNLIKELY(std::abs(tmpI[y][x] - l) > delta)) {
                    out[y][x] = get_output(y, x);
                }
            }
        };

#ifdef _OPENMP
#   pragma omp parallel if (multiThread)
#endif
    {
        for (int k = 0; k < maxiter; k++) {
            gaussianBlur(tmpI, tmp, W, H, sigma, nullptr, GAUSS_DIV, luminance);
            gaussianBlur(tmp, tmpI, W, H, sigma, nullptr, GAUSS_MULT);
#ifdef _OPENMP
#           pragma omp for
#endif
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    check_stop(y, x);
                }
            }
        }

#ifdef _OPENMP
#       pragma omp for
#endif
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float l = out[i][j];
                if (std::isnan(l)) {
                    l = get_output(i, j);
                }
                assert(std::isfinite(l));
                luminance[i][j] = std::max(l - offset, 0.f);
            }
        }
    }
#ifdef ART_USE_CUDA
    if (!gpu_result.empty()) {
        size_t different = 0, nonfinite = 0;
        double maximum = 0, sum = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float a = luminance[y][x], b = gpu_result[size_t(y) * W + x];
                nonfinite += !std::isfinite(a) || !std::isfinite(b);
                different += a != b;
                const double delta = std::abs(double(a) - b);
                maximum = std::max(maximum, delta);
                sum += delta;
            }
        }
        std::fprintf(stderr, "ART_CUDA_VERIFY width=%d height=%d sigma=%.9g different=%zu nonfinite=%zu max_abs=%.9g mean_abs=%.9g\n",
                     W, H, sigma, different, nonfinite, maximum, sum / (size_t(W) * H));
    }
#endif
}

void deconvsharpeningPair(float **center, float **corner, float **blend, char **impulse,
                         int W, int H, double sigma, double cornerSigma,
                         float amount, bool multiThread)
{
#ifdef ART_USE_CUDA
    const bool gpuPair = performanceFlag("ART_GPU_PAIR");
    if ((performanceFlag("ART_HYBRID") || gpuPair) && performanceFlag("ART_CUDA") &&
        !performanceFlag("ART_CUDA_VERIFY") && multiThread &&
        size_t(W) * H >= 8 * 1024 * 1024 && cornerSigma >= 0.25 &&
        cornerSigma <= (performanceFlag("ART_CUDA_RECURSIVE") ? 4.0 : 1.15)) {
        PipelineTimer timer(gpuPair ? "cuda_deconvolution_pair" : "hybrid_deconvolution_pair", W, H);
        std::future<bool> gpu;
        int stagingThreads = 1;
#ifdef _OPENMP
        const int originalThreads = omp_get_max_threads();
        if (!gpuPair) stagingThreads = std::min(4, std::max(1, originalThreads / 7));
        const char *workers = std::getenv("ART_HYBRID_WORKERS");
        if (workers && !gpuPair) {
            stagingThreads = std::max(1, std::min(std::atoi(workers), std::min(8, originalThreads)));
        }
#endif
        try {
            // Reserve a small CPU team for GPU staging. Packing 30 MP on one
            // core delayed the overlap; two unrestricted OMP teams contend.
            gpu = std::async(std::launch::async, [=]() {
#ifdef _OPENMP
                omp_set_num_threads(stagingThreads);
#endif
                return cuda::deconvolve(corner, blend, impulse, W, H, cornerSigma, amount, stagingThreads > 1);
            });
        } catch (const std::system_error&) {
            // No thread available: use the normal sequential dispatch below.
        } catch (const std::bad_alloc&) {
            // Allocation of the asynchronous task can also fail before launch.
        }
        if (gpu.valid()) {
            {
#ifdef _OPENMP
                struct RestoreThreads {
                    int count;
                    ~RestoreThreads() { omp_set_num_threads(count); }
                } restore{originalThreads};
                if (stagingThreads > 1) omp_set_num_threads(std::max(1, originalThreads - stagingThreads));
#endif
                deconvsharpening(center, blend, impulse, W, H, sigma, amount, multiThread, !gpuPair);
            }
            if (!gpu.get()) {
                // GPU failure leaves the corner input untouched.
                deconvsharpening(corner, blend, impulse, W, H, cornerSigma, amount, multiThread, true);
            }
            return;
        }
    }
#endif
    deconvsharpening(center, blend, impulse, W, H, sigma, amount, multiThread);
    deconvsharpening(corner, blend, impulse, W, H, cornerSigma, amount, multiThread);
}

} // namespace rtengine
