// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

namespace rtengine {
namespace cuda {

// Start context creation off the UI/processing thread when ART_CUDA=1.
// A request arriving before it completes falls back to CPU without waiting.
void warmUp();

// Keep asynchronous initialization inside main's lifetime, before CUDA and
// profiling statics are destroyed. Secondary GUI instances never start it.
void finishWarmUp();
struct WarmUpLifetime {
    ~WarmUpLifetime() { finishWarmUp(); }
};

// Experimental, opt-in (ART_CUDA=1). False leaves every input untouched so the
// caller can use the original CPU implementation. Rows need not be contiguous.
// Direct 3x3/5x5/7x7 paths; ART_CUDA_RECURSIVE=1 additionally enables the
// Young/van Vliet path through sigma=4, preserving the CPU precision tails.
bool deconvolve(float *const *luminance, float *const *blend, char *const *impulse,
                int width, int height, double sigma, float amount, bool multiThread = true);

} // namespace cuda
} // namespace rtengine
