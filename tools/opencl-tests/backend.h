// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <memory>

namespace rtengine { namespace opencl {
// Standalone experiment, NOT connected to ART's runtime dispatch.
// GPU devices only, OpenCL 1.2 API, strict FP32, bounded buffers.
// All entry points are thread-confined and share one in-order queue.
bool initialize();
bool deconvolve(float *const *lum,float *const *blend,char *const *impulse,
                int w,int h,double sigma,float amount,bool multiThread=true);
class ResizeImage {
public:
    ResizeImage();
    ~ResizeImage();
    ResizeImage(const ResizeImage&)=delete;
    ResizeImage& operator=(const ResizeImage&)=delete;
    // Always invalidates the preceding source, even if upload fails.
    bool upload(float *const *a,float *const *b,float *const *c,int w,int h);
    // Keeps destination untouched on failure; includes download and validation.
    bool resize(float *const *a,float *const *b,float *const *c,int w,int h,float scale);
    void clear();
    std::size_t deviceBytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
} }
