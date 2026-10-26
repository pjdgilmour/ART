// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../rtgui/thumbnailjobtracker.h"
#include "../../rtengine/rawframes.h"
#include <future>
#include <chrono>
#include <cstdio>
#include <stdexcept>

int main()
{
    using namespace rtengine;
    using namespace std::chrono;
    for (unsigned n : {0u, 1u, 2u, 4u, 6u, 7u, 8u}) {
        for (bool first : {false, true}) {
            auto selection = selectRawFrames(n, first);
            const unsigned count = !n ? 0 : first ? 1 : n >= 7 ? 6 : n;
            if (selection.count != count || selection.offset != (n >= 7 ? 1u : 0u)) return 1;
        }
    }
    for (int threads : {1, 2, 28}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        auto frames = selectRawFrames(4, false);
        if (loadRawFrames(frames, [](unsigned i, unsigned) {return i == 0 ? 7 : 0;}) != 7) return 2;
        if (loadRawFrames(frames, [](unsigned i, unsigned) {return i == 3 ? 8 : 0;}) != 8) return 3;
        if (loadRawFrames(selectRawFrames(2, false), [](unsigned, unsigned) {
#ifdef _OPENMP
                return omp_get_num_threads() > 2 ? 9 : 0;
#else
                return 0;
#endif
            })) return 4;
        try {
            loadRawFrames(frames, [](unsigned i, unsigned) {
                if (i == 2) throw std::runtime_error("decode failure");
                return 0;
            });
            return 5;
        } catch (const std::runtime_error &) {}
    }
    if (loadRawFrames(selectRawFrames(7, true), [](unsigned i, unsigned frame) {
            bool parallel = false;
#ifdef _OPENMP
            parallel = omp_in_parallel();
#endif
            return i == 0 && frame == 1 && !parallel ? 0 : 1;
        })) return 6;
    if (loadRawFrames(selectRawFrames(0, true), [](unsigned, unsigned) {return 0;}) == 0) return 7;

    std::mutex mutex;
    ThumbnailJobTracker active(mutex);
    int listenerA, listenerB;
    {
        std::unique_lock<std::mutex> lock(mutex);
        active.started(&listenerB);
        // Unrelated work must not hold up destruction of listener A.
        active.waitFor(lock, &listenerA);
    }
    {
        ThumbnailJobTracker::Completion completion(active, &listenerB);
    }
    {
        // Completion BEFORE entering wait: no lost wakeup.
        std::unique_lock<std::mutex> lock(mutex);
        active.waitAll(lock);
        active.started(&listenerA);
        active.started(&listenerA);
    }
    std::promise<void> entered;
    auto waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(mutex);
        entered.set_value();
        active.waitFor(lock, &listenerA);
    });
    entered.get_future().wait();
    {
        ThumbnailJobTracker::Completion first(active, &listenerA);
    }
    if (waiter.wait_for(milliseconds(20)) == std::future_status::ready) return 8;
    try {
        ThumbnailJobTracker::Completion second(active, &listenerA);
        throw std::runtime_error("thumbnail failure");
    } catch (const std::runtime_error &) {}
    if (waiter.wait_for(seconds(2)) != std::future_status::ready) return 9;
    waiter.get();
    std::puts("PASS RAW frame selection/errors/exceptions/thread budget; thumbnail completion, scoped waits and exception cleanup");
}
