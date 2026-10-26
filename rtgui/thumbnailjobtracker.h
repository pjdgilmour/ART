// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <condition_variable>
#include <map>
#include <mutex>

// The caller's mutex also protects the pending queue: removing queued work,
// registering active work and testing the wait predicate must be atomic.
class ThumbnailJobTracker {
public:
    explicit ThumbnailJobTracker(std::mutex &mutex): mutex_(mutex) {}

    void started(const void *listener) // caller holds mutex_
    {
        ++active_[listener];
    }

    void waitFor(std::unique_lock<std::mutex> &lock, const void *listener)
    {
        changed_.wait(lock, [&] { return active_.find(listener) == active_.end(); });
    }

    void waitAll(std::unique_lock<std::mutex> &lock)
    {
        changed_.wait(lock, [&] { return active_.empty(); });
    }

    class Completion {
    public:
        Completion(ThumbnailJobTracker &owner, const void *listener):
            owner_(owner), listener_(listener) {}
        ~Completion() { owner_.finished(listener_); }
        Completion(const Completion &) = delete;
        Completion &operator=(const Completion &) = delete;
    private:
        ThumbnailJobTracker &owner_;
        const void *listener_;
    };

private:
    void finished(const void *listener)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto entry = active_.find(listener);
            if (--entry->second == 0) {
                active_.erase(entry);
            }
            // A waiter may destroy the tracker as soon as it reacquires the
            // lock and sees no active jobs. Notify before releasing that lock.
            changed_.notify_all();
        }
    }

    std::mutex &mutex_;
    std::condition_variable changed_;
    std::map<const void *, unsigned> active_;
};
