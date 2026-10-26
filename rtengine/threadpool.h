// -*- C++ -*-
//
// Adapted from https://github.com/progschj/ThreadPool
/*
Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/

#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <limits>

#include "noncopyable.h"
#include "cpupolicy.h"

namespace rtengine {

class ThreadPool: public NonCopyable {
public: 
    enum class Priority {
        LOWEST,
        LOW,
        NORMAL,
        HIGH,
        HIGHEST
    };

    template<class F, class... Args>
    static auto add_task(Priority p, F &&f, Args &&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    static void init(size_t num_workers);
    static void cleanup();
    struct Snapshot { size_t active=0, queued=0, background=0; };
    static Snapshot snapshot() {
        if (!instance_) return Snapshot();
        std::lock_guard<std::mutex> lock(instance_->queue_mutex_);
        Snapshot s; s.active=instance_->active_;s.queued=instance_->tasks_.size();s.background=instance_->background_active_;return s;
    }
    class Interactive {
        bool active_;
    public:
        Interactive():active_(adaptiveScheduling() && bool(instance_)) {if(active_) changeInteractive(1);}
        ~Interactive() {if(active_) changeInteractive(-1);}
        Interactive(const Interactive&)=delete;
        Interactive& operator=(const Interactive&)=delete;
    };
    static void changeInteractive(int delta) {
        if(!instance_) return;
        {std::lock_guard<std::mutex> lock(instance_->queue_mutex_);instance_->interactive_+=delta;}
        instance_->condition_.notify_all();
    }
    
private:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(Priority p, F &&f, Args &&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    template<class F, class... Args>
    auto enqueue(F &&f, Args &&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        return enqueue(Priority::NORMAL, f, args...);
    }

public:
    ~ThreadPool();

private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers_;
    // the task queue
    class Task: public std::function<void()> {
    public:
        template <class Fn>
        explicit Task(Priority p, size_t i, Fn func):
            std::function<void()>(func), p_(p), i_(i) {}

        bool background() const { return p_ < Priority::NORMAL; }
        bool foreground() const { return p_ > Priority::NORMAL; }
        bool operator<(const Task &other) const
        {
            return p_ == other.p_ ? i_ > other.i_ : p_ < other.p_;
        }
    private:
        Priority p_;
        size_t i_;
    };
    std::priority_queue<Task> tasks_;
    
    // synchronization
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
    size_t idx_;
    size_t active_=0, background_active_=0;
    int interactive_=0;

    static std::unique_ptr<ThreadPool> instance_;
};


// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads):
    stop_(false),
    idx_(0)
{
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back(
            [this]
            {
                while (true) {
                    std::function<void()> task;
                    bool background=false, foreground=false;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(
                            lock,
                            [this]{
                                if(this->stop_) return true;
                                if(this->tasks_.empty()) return false;
                                if(!adaptiveScheduling() || !this->tasks_.top().background()) return true;
                                const size_t limit=this->interactive_>0?1:backgroundBudget();
                                return this->background_active_<limit;
                            });
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }
                        background=this->tasks_.top().background();
                        foreground=this->tasks_.top().foreground();
                        ++active_;if(background) ++background_active_;if(foreground) ++interactive_;
                        task = std::move(this->tasks_.top());
                        this->tasks_.pop();
                    }

                    applyCpuPolicy(background);
                    task();
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        --active_;if(background) --background_active_;if(foreground) --interactive_;
                    }
                    condition_.notify_all();
                }
            }
        );
    }
}


// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(Priority p, F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // don't allow enqueueing after stopping the pool
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks_.emplace(p, idx_++, [task](){ (*task)(); });
    }
    condition_.notify_all();
    return res;
}


// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker: workers_) {
        worker.join();
    }
}


inline void ThreadPool::init(size_t num_workers)
{
    instance_.reset(new ThreadPool(num_workers));
}


inline void ThreadPool::cleanup()
{
    instance_.reset(nullptr);
}


template<class F, class... Args>
auto ThreadPool::add_task(Priority p, F &&f, Args &&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    return instance_->enqueue(p, f, args...);
}

} // namespace rtengine

