// SPDX-License-Identifier: GPL-3.0-or-later
// CUDA port of the separable Lanczos loop in ART/RawTherapee ipresize.cc.
// Original resampler: Copyright (c) 2004-2010 Gabor Horvath.
#include "resize.h"
#include "../lanczos.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

namespace rtengine { namespace cuda {
namespace {
constexpr size_t budget = size_t(1024) * 1024 * 1024;
constexpr size_t maxPixels = size_t(64) * 1024 * 1024;
bool dimensions(int w, int h) { return w > 0 && h > 0 && size_t(w)*h <= maxPixels; }
__global__ void vertical(const float *src, float *tmp, int sw, int sh, int dh,
                         const float *weights, const int *first, const int *last, int support)
{
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= size_t(sw)*dh*3) return;
    int x = i % sw, y = (i/sw) % dh, c = i/(size_t(sw)*dh);
    float v = 0;
    for (int k = first[y]; k < last[y]; ++k)
        v += weights[size_t(y)*support+k-first[y]] * src[(size_t(c)*sh+k)*sw+x];
    tmp[i] = v;
}
__global__ void horizontal(const float *tmp, float *out, int sw, int dw, int dh,
                           const float *weights, const int *first, const int *last, int support)
{
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= size_t(dw)*dh*3) return;
    int x = i % dw, y = (i/dw) % dh, c = i/(size_t(dw)*dh);
    float v = 0;
    for (int k = first[x]; k < last[x]; ++k)
        v += weights[size_t(x)*support+k-first[x]] * tmp[(size_t(c)*dh+y)*sw+k];
    out[i] = v;
}
}
struct ResizeImage::Impl {
    float *source = nullptr, *scratch = nullptr;
    size_t scratchBytes = 0;
    int width = 0, height = 0;
    cudaStream_t stream = nullptr;
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(scratch); cudaFree(source);
        if (stream) cudaStreamDestroy(stream);
    }
};
ResizeImage::ResizeImage() = default;
ResizeImage::~ResizeImage() = default;
void ResizeImage::clear() { p.reset(); }
size_t ResizeImage::deviceBytes() const {
    return p ? size_t(p->width)*p->height*3*sizeof(float)+p->scratchBytes : 0;
}
bool ResizeImage::upload(float *const *a, float *const *b, float *const *c, int w, int h)
{
    clear();
    if (!dimensions(w,h) || !a || !b || !c) return false;
    try {
        const size_t n = size_t(w)*h;
        std::unique_ptr<float[]> host(new float[n*3]);
        float *const *planes[] = {a,b,c};
        int finite = 1;
#ifdef ART_CUDA_OPENMP
        #pragma omp parallel for reduction(&:finite)
#endif
        for (int y=0; y<h; ++y) for (int ch=0; ch<3; ++ch) {
            for (int x=0; x<w; ++x) finite &= std::isfinite(planes[ch][y][x]);
            std::memcpy(host.get()+ch*n+size_t(y)*w, planes[ch][y], size_t(w)*sizeof(float));
        }
        if (!finite) return false;
        p.reset(new Impl);
        if (cudaStreamCreateWithFlags(&p->stream,cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc(&p->source,n*3*sizeof(float)) != cudaSuccess ||
            cudaMemcpyAsync(p->source,host.get(),n*3*sizeof(float),cudaMemcpyHostToDevice,p->stream) != cudaSuccess ||
            cudaStreamSynchronize(p->stream) != cudaSuccess) { clear(); return false; }
        p->width=w; p->height=h;
        return true;
    } catch (const std::bad_alloc&) { clear(); return false; }
}
bool ResizeImage::resize(float *const *a, float *const *b, float *const *c, int w, int h, float scale)
{
    if (!p || !p->width || !dimensions(w,h) || !a || !b || !c ||
        !std::isfinite(scale) || scale < 0.05f || scale > 4.f ||
        w > std::ceil(p->width*scale) || h > std::ceil(p->height*scale)) return false;
    try {
        const int support = int(6.f/std::min(scale,1.f))+1;
        const size_t n = size_t(w)*h*3, mid = size_t(p->width)*h*3;
        const size_t nw = size_t(w+h)*support;
        const size_t bytes = (mid+n+nw)*sizeof(float)+size_t(w+h)*2*sizeof(int);
        if (bytes+size_t(p->width)*p->height*3*sizeof(float) > budget) return false;
        std::vector<float> weights(nw,0.f);
        std::vector<int> first(w+h),last(w+h);
        lanczosWeights(p->width,w,scale,support,weights.data(),first.data(),last.data());
        lanczosWeights(p->height,h,scale,support,weights.data()+size_t(w)*support,first.data()+w,last.data()+w);
        for (int j=0; j<w+h; ++j) if (first[j]>=last[j]) return false;
        if (bytes > p->scratchBytes) {
            cudaFree(p->scratch); p->scratch=nullptr; p->scratchBytes=0;
            if (cudaMalloc(&p->scratch,bytes) != cudaSuccess) { clear(); return false; }
            p->scratchBytes=bytes;
        }
        auto tmp=p->scratch, out=tmp+mid, ww=out+n;
        int *ff=reinterpret_cast<int*>(ww+nw), *ll=ff+w+h;
        std::unique_ptr<float[]> host(new float[n]);
        auto copy=[&](void *d,const void *s,size_t count) {
            return cudaMemcpyAsync(d,s,count,cudaMemcpyHostToDevice,p->stream)==cudaSuccess;
        };
        if (!copy(ww,weights.data(),nw*sizeof(float)) || !copy(ff,first.data(),(w+h)*sizeof(int)) ||
            !copy(ll,last.data(),(w+h)*sizeof(int))) { clear(); return false; }
        vertical<<<(mid+255)/256,256,0,p->stream>>>(p->source,tmp,p->width,p->height,h,
                                                ww+size_t(w)*support,ff+w,ll+w,support);
        if (cudaGetLastError()!=cudaSuccess) { clear(); return false; }
        horizontal<<<(n+255)/256,256,0,p->stream>>>(tmp,out,p->width,w,h,ww,ff,ll,support);
        if (cudaGetLastError()!=cudaSuccess ||
            cudaMemcpyAsync(host.get(),out,n*sizeof(float),cudaMemcpyDeviceToHost,p->stream)!=cudaSuccess ||
            cudaStreamSynchronize(p->stream)!=cudaSuccess) { clear(); return false; }
        int finite=1;
#ifdef ART_CUDA_OPENMP
        #pragma omp parallel for reduction(&:finite)
#endif
        for (size_t i=0; i<n; ++i) finite &= std::isfinite(host[i]);
        if (!finite) { clear(); return false; }
        float *const *planes[]={a,b,c};
#ifdef ART_CUDA_OPENMP
        #pragma omp parallel for
#endif
        for (int y=0; y<h; ++y) for (int ch=0; ch<3; ++ch)
            std::memcpy(planes[ch][y],host.get()+size_t(ch)*w*h+size_t(y)*w,size_t(w)*sizeof(float));
        return true;
    } catch (const std::bad_alloc&) { clear(); return false; }
}
} }
