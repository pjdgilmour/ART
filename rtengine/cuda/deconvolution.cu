/*
 * Selective CUDA port of ART's deconvsharpening and direct Gaussian kernels.
 * Derived from rtengine/ipsharpen.cc and rtengine/gauss.cc, part of RawTherapee/ART.
 * Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Keep arithmetic order, border handling and stopping rules in sync with the
 * CPU reference. This is deliberately not a replacement Gaussian algorithm.
 */
#include "deconvolution.h"
#include "../pipelinetime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>
#include <thread>
#include <system_error>

namespace rtengine {
namespace cuda {
namespace {

#include "recursive_gaussian.cuh"

struct Coefficients {
    float c00, c10, c11, c20, c21, c22, c30, c31, b0, b1;
};

Coefficients coefficients(double sigma, int radius)
{
    Coefficients c{};
    if (radius == 1) {
        const double inv = 1.0 / sigma;
        double c1 = std::exp(-0.5 * (inv * inv));
        double c2 = std::exp(-(inv * inv));
        const double sum = 1.0 + 4.0 * (c1 + c2);
        c.c00 = 1.0 / sum; c.c10 = c1 / sum; c.c11 = c2 / sum;
        double b1 = std::exp(-1.0 / (2.0 * sigma * sigma));
        const double bsum = 2.0 * b1 + 1.0;
        c.b0 = 1.0 / bsum; c.b1 = b1 / bsum;
        return c;
    }
    // Exactly the float accumulation and truncated support of compute5x5kernel
    // / compute7x7kernel in gauss.cc (including conversion of sigma to float).
    const float sf = sigma;
    const double temp = -2.f * (sf * sf);
    const double cutoff = 3.0 * (radius == 2 ? 0.84 : 1.15);
    float k[7][7] = {};
    float sum = 0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= cutoff * cutoff) {
                k[y + 3][x + 3] = std::exp((x * x + y * y) / temp);
                sum += k[y + 3][x + 3];
            }
        }
    }
    c.c00 = k[3][3] / sum; c.c10 = k[2][3] / sum;
    c.c11 = k[2][2] / sum; c.c20 = k[1][3] / sum;
    c.c21 = k[1][2] / sum; c.c22 = k[1][1] / sum;
    c.c30 = k[0][3] / sum; c.c31 = k[0][2] / sum;
    return c;
}

template<int R>
__device__ float gaussian(const float *s, int x, int y, int w, int h, Coefficients c)
{
    const int i = y * w + x;
    if (R == 1) {
        if (y == 0 || y == h - 1) {
            if (x == 0 || x == w - 1) return s[i];
            return c.b1 * (s[i-1] + s[i+1]) + c.b0 * s[i];
        }
        if (x == 0 || x == w - 1) return c.b1 * (s[i-w] + s[i+w]) + c.b0 * s[i];
        return c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    if (R == 2) {
        return c.c21 * (s[i-2*w-1] + s[i-2*w+1] + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
               c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
               c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    // Preserve the existing CPU's extra c21 factor on this one term. Changing
    // it would change saved edits; any correction belongs in a separate change.
    return c.c31 * (s[i-3*w-1] + s[i-3*w+1] + s[i-w-3] + s[i-w+3] + s[i+w-3] + s[i+w+3] + s[i+3*w-1] + s[i+3*w+1]) +
           c.c30 * (s[i-3*w] + s[i-3] + s[i+3] + s[i+3*w]) +
           c.c22 * (s[i-2*w-2] + s[i-2*w+2] + s[i+2*w-2] + s[i+2*w+2]) +
           c.c21 * (s[i-2*w-1] + s[i-2*w+1] * c.c21 + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
           c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
           c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
           c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
}

__global__ void initialize(float *lum, float *estimate, float *stopped, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        lum[i] += 1000.f;
        estimate[i] = fmaxf(lum[i], 0.f);
        stopped[i] = nanf("");
    }
}

__device__ float output(float original, float estimate, float blend, unsigned char impulse, float amount)
{
    if (isnan(estimate)) return original;
    const float b = impulse ? 0.f : blend * amount;
    return b * fmaxf(estimate, 0.f) + (1.f - b) * original;
}

template<int R>
__global__ void divide(const float *estimate, const float *lum, float *ratio, int w, int h, Coefficients c)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    if (R > 1 && (x < R || y < R || x >= w-R || y >= h-R)) {
        ratio[i] = 1.f;
        return;
    }
    const float v = gaussian<R>(estimate, x, y, w, h, c);
    ratio[i] = R == 1 ? fmaxf(lum[i] / (v > 0.f ? v : 1.f), 0.f) : lum[i] / fmaxf(v, 0.00001f);
}

template<int R>
__global__ void multiply(const float *ratio, const float *lum, const float *blend, const unsigned char *impulse,
                         float *estimate, float *stopped, int w, int h, float amount, Coefficients c)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    float v = estimate[i];
    if (R == 1 || (x >= R && y >= R && x < w-R && y < h-R)) {
        v *= gaussian<R>(ratio, x, y, w, h, c);
        estimate[i] = v;
    }
    if (isnan(stopped[i]) && fabsf(v - lum[i]) > lum[i] * 0.2f) {
        stopped[i] = output(lum[i], v, blend[i], impulse[i], amount);
    }
}

__global__ void finish(const float *lum, const float *blend, const unsigned char *impulse,
                       const float *estimate, const float *stopped, float *result, int n, float amount)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = stopped[i];
        if (isnan(v)) v = output(lum[i], estimate[i], blend[i], impulse[i], amount);
        result[i] = fmaxf(v - 1000.f, 0.f);
    }
}

template<class T> struct HostDelete {
    bool pinned;
    HostDelete(bool value = false): pinned(value) {}
    void operator()(T *ptr) const {
        if (pinned) cudaFreeHost(ptr);
        else delete[] ptr;
    }
};
template<class T> using HostBuffer = std::unique_ptr<T[], HostDelete<T>>;
template<class T> HostBuffer<T> allocateHost(size_t n, bool pinned) {
    if (pinned) {
        T *ptr = nullptr;
        if (cudaMallocHost(&ptr, n * sizeof(T)) == cudaSuccess) {
            return HostBuffer<T>(ptr, HostDelete<T>(true));
        }
        cudaGetLastError(); // Pageable staging remains a valid fallback.
    }
    return HostBuffer<T>(new T[n]);
}

struct Context {
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    float *device = nullptr;
    unsigned char *impulses = nullptr;
    size_t capacity = 0;
    float *recursive_scratch = nullptr;
    size_t recursive_capacity = 0;
    bool failed = false;
    bool warming_started = false;
    bool ready = false;
    std::thread warmer;
    HostBuffer<float> host;
    HostBuffer<unsigned char> host_impulse;

    ~Context()
    {
        if (warmer.joinable()) warmer.join();
        if (stream) cudaStreamSynchronize(stream);
        if (device) cudaFree(device);
        if (impulses) cudaFree(impulses);
        if (recursive_scratch) cudaFree(recursive_scratch);
        if (stream) cudaStreamDestroy(stream);
    }

    void check(cudaError_t error)
    {
        if (error != cudaSuccess) throw error;
    }

    void prepare()
    {
        check(cudaSetDevice(0));
        if (!stream) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ready = true;
    }

    void reserve(size_t n)
    {
        prepare();
        if (n <= capacity) return;
        // All earlier work is complete while holding mutex.
        if (device) { check(cudaFree(device)); device = nullptr; }
        if (impulses) { check(cudaFree(impulses)); impulses = nullptr; }
        capacity = 0;
        const char *pinned_env = std::getenv("ART_CUDA_PINNED");
        const bool pinned = pinned_env && std::strcmp(pinned_env, "1") == 0;
        host = allocateHost<float>(n * 2, pinned);
        host_impulse = allocateHost<unsigned char>(n, pinned);
        check(cudaMalloc(&device, n * 5 * sizeof(float)));
        check(cudaMalloc(&impulses, n));
        capacity = n;
    }
};

__global__ void recursiveDivide(const float *blur,const float *lum,float *ratio,int w,int h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    float v=lum[i]/(blur[i]>0.f?blur[i]:1.f);
    // The existing SSE vertical divide leaves its last three rows unclamped.
    ratio[i]=(i/w<h-3 || i%w>=w-w%8) ? fmaxf(v,0.f) : v;
}
__global__ void recursiveMultiply(const float *blur,const float *lum,const float *blend,
                                  const unsigned char *impulse,float *estimate,float *stopped,int n,float amount) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float v=estimate[i]*blur[i];estimate[i]=v;
    if(isnan(stopped[i]) && fabsf(v-lum[i])>lum[i]*.2f)
        stopped[i]=output(lum[i],v,blend[i],impulse[i],amount);
}

void recursiveIterations(Context &ctx,int w,int h,double sigma,float amount) {
    const int n=w*h,blocks=(n+255)/256;
    if(size_t(n)>ctx.recursive_capacity) {
        if(ctx.recursive_scratch) { ctx.check(cudaFree(ctx.recursive_scratch));ctx.recursive_scratch=nullptr; }
        ctx.recursive_capacity=0;
        ctx.check(cudaMalloc(&ctx.recursive_scratch,size_t(n)*2*sizeof(float)));
        ctx.recursive_capacity=n;
    }
    float *lum=ctx.device,*blend=lum+n,*estimate=blend+n,*ratio=estimate+n,*stopped=ratio+n;
    auto c=recursive::factors(sigma);
    initialize<<<blocks,256,0,ctx.stream>>>(lum,estimate,stopped,n);
    for(int k=0;k<20;++k) {
        recursive::blur(estimate,ctx.recursive_scratch,ctx.recursive_scratch+n,w,h,c,ctx.stream);
        recursiveDivide<<<blocks,256,0,ctx.stream>>>(ctx.recursive_scratch+n,lum,ratio,w,h);
        recursive::blur(ratio,ctx.recursive_scratch,ctx.recursive_scratch+n,w,h,c,ctx.stream);
        recursiveMultiply<<<blocks,256,0,ctx.stream>>>(ctx.recursive_scratch+n,lum,blend,ctx.impulses,estimate,stopped,n,amount);
    }
    finish<<<blocks,256,0,ctx.stream>>>(lum,blend,ctx.impulses,estimate,stopped,ratio,n,amount);
    ctx.check(cudaGetLastError());
    ctx.check(cudaMemcpyAsync(ctx.host.get(),ratio,size_t(n)*sizeof(float),cudaMemcpyDeviceToHost,ctx.stream));
    ctx.check(cudaStreamSynchronize(ctx.stream));
}

Context &context(int index = 0)
{
    static Context contexts[2];
    return contexts[index];
}

bool enabled()
{
    const char *value = std::getenv("ART_CUDA");
    return value && std::strcmp(value, "1") == 0;
}

template<int R>
void iterations(Context &ctx, int w, int h, float amount, Coefficients c)
{
    const int n = w * h;
    float *lum = ctx.device, *blend = lum + n, *estimate = blend + n;
    float *ratio = estimate + n, *stopped = ratio + n;
    const int blocks = (n + 255) / 256;
    initialize<<<blocks, 256, 0, ctx.stream>>>(lum, estimate, stopped, n);
    for (int k = 0; k < 20; ++k) {
        divide<R><<<blocks, 256, 0, ctx.stream>>>(estimate, lum, ratio, w, h, c);
        multiply<R><<<blocks, 256, 0, ctx.stream>>>(ratio, lum, blend, ctx.impulses, estimate, stopped, w, h, amount, c);
    }
    finish<<<blocks, 256, 0, ctx.stream>>>(lum, blend, ctx.impulses, estimate, stopped, ratio, n, amount);
    ctx.check(cudaGetLastError());
    ctx.check(cudaMemcpyAsync(ctx.host.get(), ratio, n * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.check(cudaStreamSynchronize(ctx.stream));
}

} // namespace

void warmUp()
{
    if (!enabled()) return;
    auto &ctx = context();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    if (ctx.warming_started || ctx.failed) return;
    ctx.warming_started = true;
    try {
        ctx.warmer = std::thread([&ctx] {
            std::lock_guard<std::mutex> lock(ctx.mutex);
            PipelineTimer timer("cuda_context_init");
            try {
                ctx.prepare();
            } catch (cudaError_t error) {
                ctx.failed = true;
                std::fprintf(stderr, "ART CUDA: initialization failed, using CPU: %s\n", cudaGetErrorString(error));
            }
        });
    } catch (const std::system_error&) {
        ctx.failed = true;
        std::fprintf(stderr, "ART CUDA: initialization worker unavailable, using CPU\n");
    }
}

void finishWarmUp()
{
    // Called by the owning main thread after all warmUp() calls. Do not hold
    // ctx.mutex here: initialization itself needs it.
    auto &ctx = context();
    if (ctx.warmer.joinable()) ctx.warmer.join();
}

bool deconvolve(float *const *luminance, float *const *blend, char *const *impulse,
                int w, int h, double sigma, float amount, bool multiThread)
{
    const char *recursive_env=std::getenv("ART_CUDA_RECURSIVE");
    const bool allow_recursive=recursive_env && std::strcmp(recursive_env,"1")==0;
    // Bounded experimental backend domain. The engine separately applies the
    // measured recursive size threshold. Unset ART_CUDA never starts CUDA.
    if (!enabled() || w < 8 || h < 8 ||
        size_t(w) * h < 1024 * 1024 || size_t(w) * h > 64 * 1024 * 1024 ||
        !std::isfinite(sigma) || sigma < 0.25 || sigma > (allow_recursive ? 4.0 : 1.15) ||
        !std::isfinite(amount) || amount <= 0) return false;

    Context *selected = &context();
    std::unique_lock<std::mutex> lock(selected->mutex, std::try_to_lock);
    const char *pair_env = std::getenv("ART_GPU_PAIR");
    if (!lock.owns_lock() && pair_env && std::strcmp(pair_env, "1") == 0) {
        selected = &context(1);
        lock = std::unique_lock<std::mutex>(selected->mutex, std::try_to_lock);
    }
    if (!lock.owns_lock() || selected->failed || (selected->warming_started && !selected->ready)) return false;
    auto &ctx = *selected;
    PipelineTimer timer("cuda_deconvolution_attempt", w, h);
    const size_t n = size_t(w) * h;
    try {
        {
            PipelineTimer timer("cuda_prepare_buffers", w, h);
            ctx.reserve(n);
        }
        {
            PipelineTimer timer("cuda_pack", w, h);
            int invalid = 0;
#ifdef ART_CUDA_OPENMP
#pragma omp parallel for if(multiThread) reduction(|:invalid) schedule(static)
#endif
            for (int y = 0; y < h; ++y) {
                const size_t row = size_t(y) * w;
                for (int x = 0; x < w; ++x) {
                    const float l = luminance[y][x], b = blend[y][x];
                    ctx.host[row + x] = l;
                    ctx.host[n + row + x] = b;
                    ctx.host_impulse[row + x] = impulse[y][x];
                    invalid |= !std::isfinite(l) || !std::isfinite(b);
                }
            }
            if (invalid) return false;
        }
        {
            PipelineTimer timer("cuda_transfer_and_compute", w, h);
            ctx.check(cudaMemcpyAsync(ctx.device, ctx.host.get(), n * 2 * sizeof(float), cudaMemcpyHostToDevice, ctx.stream));
            ctx.check(cudaMemcpyAsync(ctx.impulses, ctx.host_impulse.get(), n, cudaMemcpyHostToDevice, ctx.stream));
            const int radius = sigma < 0.6 ? 1 : sigma <= 0.84 ? 2 : 3;
            const auto c = coefficients(sigma, radius);
            if (sigma > 1.15) recursiveIterations(ctx,w,h,sigma,amount);
            else if (radius == 1) iterations<1>(ctx, w, h, amount, c);
            else if (radius == 2) iterations<2>(ctx, w, h, amount, c);
            else iterations<3>(ctx, w, h, amount, c);
        }
        PipelineTimer commit_timer("cuda_commit", w, h);
        // Commit only after the entire result has been downloaded and checked.
        int invalid = 0;
#ifdef ART_CUDA_OPENMP
#pragma omp parallel for if(multiThread) reduction(|:invalid) schedule(static)
#endif
        for (int i = 0; i < int(n); ++i) {
            invalid |= !std::isfinite(ctx.host[i]);
        }
        if (invalid) return false;
#ifdef ART_CUDA_OPENMP
#pragma omp parallel for if(multiThread) schedule(static)
#endif
        for (int y = 0; y < h; ++y) {
            std::memcpy(luminance[y], ctx.host.get() + size_t(y) * w, w * sizeof(float));
        }
        return true;
    } catch (cudaError_t error) {
        if (ctx.stream) cudaStreamSynchronize(ctx.stream);
        ctx.failed = true;
        std::fprintf(stderr, "ART CUDA: deconvolution disabled, using CPU: %s\n", cudaGetErrorString(error));
    } catch (const std::bad_alloc&) {
        ctx.failed = true;
        std::fprintf(stderr, "ART CUDA: staging allocation failed, using CPU\n");
    }
    return false;
}

} // namespace cuda
} // namespace rtengine
