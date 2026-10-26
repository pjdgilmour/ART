// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <memory>

namespace rtengine { namespace cuda {
// Experimental, explicitly owned cache for ONE decoded planar image, <= 1 GiB.
// Thread-confined. Call upload after EVERY source/processing/profile change.
// upload invalidates the old image even on failure; pointer identity is NOT a key.
// resize never modifies the destination on failure. clear releases all VRAM.
class ResizeImage {
public:
    ResizeImage();
    ~ResizeImage();
    ResizeImage(const ResizeImage&) = delete;
    ResizeImage& operator=(const ResizeImage&) = delete;
    bool upload(float *const *a, float *const *b, float *const *c, int w, int h);
    bool resize(float *const *a, float *const *b, float *const *c,
                int w, int h, float scale);
    void clear();
    std::size_t deviceBytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
} }
