// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../rtengine/threadpool.h"
#include "../../rtengine/rawconvert.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <atomic>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace rtengine {std::unique_ptr<ThreadPool> ThreadPool::instance_;}
int main() {
    using namespace rtengine;
    const auto &t=cpuTopology();
    std::printf("logical=%u physical=%u sockets=%u topology_known=%d avx2=%d fma=%d avx512=%d processing=%d background=%d\n",t.logical,t.physical,t.sockets,t.known,t.avx2,t.fma,t.avx512,processingBudget(),backgroundBudget());
    std::printf("numa=%u largest_cache_kib=%u line=%u capacity_classes=%u quota=%u\n",t.numaNodes,t.largestCacheKiB,t.cacheLine,t.capacityClasses,t.quotaThreads);
    if(processingBudget()<1 || processingBudget()>int(t.logical)) return 1;
    for(int w:{1,2,7,8,9,31,1025,6732}) for(int a=0;a<4;++a) for(int b=0;b<4;++b) {
        std::vector<uint16_t> input(4*w);
        std::vector<float> scalar(w+1,-123.f),simd(w+1,-123.f);
        for(size_t i=0;i<input.size();++i) input[i]=(i*1664525+1013904223)&65535;
        unpackBayerRow(input.data(),scalar.data(),w,a,b,false);
        unpackBayerRow(input.data(),simd.data(),w,a,b,true);
        if(scalar!=simd || simd[w]!=-123.f) return 2;
        for(int x=0;x<w;++x) if(simd[x]!=float(input[4*x+((x&1)?b:a)])) return 3;
    }
    if(adaptiveScheduling()) {
        ThreadPool::init(backgroundBudget()+1);
        std::promise<void> release;auto signal=release.get_future().share();
        std::atomic<int> started(0);
        std::vector<std::future<void>> background;
        for(int i=0;i<backgroundBudget();++i) background.push_back(ThreadPool::add_task(ThreadPool::Priority::LOW,[&] {++started;signal.wait();}));
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(started<backgroundBudget() && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        auto foreground=ThreadPool::add_task(ThreadPool::Priority::HIGHEST,[] {return 42;});
        const bool ready=foreground.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        release.set_value();for(auto &f:background) f.get();
        ThreadPool::cleanup();
        if(!ready || foreground.get()!=42) return 4;
    }
    std::puts("PASS topology budget, scalar/AVX2 exact conversion, tails and foreground capacity");
}
