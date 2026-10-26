// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtengine {
struct RawFrameSelection {
    unsigned count;
    unsigned offset;
};

inline RawFrameSelection selectRawFrames(unsigned available, bool firstOnly)
{
    // Preserve the existing multi-shot rule: the first frame is not Bayer.
    const unsigned offset = available >= 7 ? 1 : 0;
    return {available ? (firstOnly ? 1u : std::min(6u, available - offset)) : 0u, offset};
}

template<class Loader>
int loadRawFrames(RawFrameSelection frames, Loader load)
{
    if (!frames.count || frames.count > 6) {
        return 2;
    }
    std::array<int, 6> errors{};
    std::array<std::exception_ptr, 6> exceptions{};
    const auto decode = [&](unsigned index) {
        try {
            errors[index] = load(index, index + frames.offset);
        } catch (...) {
            // Exceptions may not escape an OpenMP region. Propagate after join
            // so the image source's owner can clean up every partial frame.
            exceptions[index] = std::current_exception();
        }
    };
    if (frames.offset == 0 && frames.count > 1) {
#ifdef _OPENMP
        const int threads = std::min<int>(frames.count, omp_get_max_threads());
        #pragma omp parallel for num_threads(threads)
#endif
        for (unsigned i = 0; i < frames.count; ++i) {
            decode(i);
        }
    } else {
        // Keep the special multi-shot format on its original serial path.
        for (unsigned i = 0; i < frames.count; ++i) {
            decode(i);
        }
    }
    for (unsigned i = 0; i < frames.count; ++i) {
        if (exceptions[i]) {
            std::rethrow_exception(exceptions[i]);
        }
        if (errors[i]) {
            return errors[i];
        }
    }
    return 0;
}
} // namespace rtengine
