// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <gdk/gdk.h>
#include <cstdint>
#include <memory>

// One monitor-converted RGB8 image per view, owned by the GTK main thread.
// The caller MUST change generation when pixels or their color transform change.
// No CUDA dependency: presentation uses the GL backend selected by GDK.
class GpuPreview {
public:
    GpuPreview();
    ~GpuPreview();
    GpuPreview(const GpuPreview&) = delete;
    GpuPreview& operator=(const GpuPreview&) = delete;
    bool draw(cairo_t *cr, GdkWindow *window, GdkPixbuf *pixels, uint64_t generation,
              int x, int y, int width, int height);
    void clear();
    // Diagnostics/test: verify the cached RGB bytes, including orientation.
    bool matches(GdkPixbuf *pixels);
    size_t deviceBytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
