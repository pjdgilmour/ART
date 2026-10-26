// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpupreview.h"
#include <epoxy/gl.h>
#include <cstring>
#include <new>
#include <vector>
#include "../../rtengine/pipelinetime.h"

namespace {
struct CurrentContext {
    GdkGLContext *saved = gdk_gl_context_get_current();
    CurrentContext() { if(saved) g_object_ref(saved); }
    ~CurrentContext() {
        if(saved) { gdk_gl_context_make_current(saved); g_object_unref(saved); }
        else gdk_gl_context_clear_current();
    }
};
}
struct GpuPreview::Impl {
    GdkGLContext *context=nullptr;
    GdkWindow *window=nullptr;
    GLuint texture=0;
    uint64_t generation=0;
    GdkPixbuf *identity=nullptr; // compared only; generation is the validity key
    int width=0,height=0;
    bool valid=false,failed=false;
    ~Impl() {
        CurrentContext restore;
        if(context) {
            gdk_gl_context_make_current(context);
            if(texture) glDeleteTextures(1,&texture);
            g_object_unref(context);
        }
        if(window) g_object_unref(window);
    }
};
GpuPreview::GpuPreview() = default;
GpuPreview::~GpuPreview() = default;
void GpuPreview::clear() { p.reset(); }
size_t GpuPreview::deviceBytes() const {
    // Conservative accounting: drivers can store RGB8 as RGBA8.
    return p && p->valid ? size_t(p->width)*p->height*4 : 0;
}
bool GpuPreview::draw(cairo_t *cr,GdkWindow *window,GdkPixbuf *pixels,uint64_t generation,
                      int x,int y,int width,int height)
{
    if(!window || gdk_window_is_destroyed(window) || gdk_window_get_scale_factor(window)!=1 ||
       !pixels || width<=0 || height<=0 ||
       gdk_pixbuf_get_n_channels(pixels)!=3 || gdk_pixbuf_get_bits_per_sample(pixels)!=8) return false;
    const int w=gdk_pixbuf_get_width(pixels),h=gdk_pixbuf_get_height(pixels);
    if(w<=0 || h<=0 || size_t(w)*h*4>size_t(256)*1024*1024 || width>w || height>h) return false;
    if(p && p->window!=window) clear();
    if(p && p->failed) return false;
    CurrentContext restore;
    try {
        if(!p) p.reset(new Impl);
        if(!p->context) {
            p->window=GDK_WINDOW(g_object_ref(window));
            GError *error=nullptr;
            p->context=gdk_window_create_gl_context(window,&error);
            if(p->context) gdk_gl_context_set_use_es(p->context,FALSE);
            if(!p->context || !gdk_gl_context_realize(p->context,&error)) {
                if(error) { g_warning("ART GL preview: %s",error->message); g_error_free(error); }
                p->failed=true; return false;
            }
            gdk_gl_context_make_current(p->context);
            g_message("ART GL preview: %s",glGetString(GL_RENDERER));
            glGenTextures(1,&p->texture);
        }
        gdk_gl_context_make_current(p->context);
        GLint maxSize=0; glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxSize);
        if(w>maxSize || h>maxSize) return false;
        if(!p->valid || p->generation!=generation || p->identity!=pixels || p->width!=w || p->height!=h) {
            rtengine::PipelineTimer timer("gl_preview_upload",w,h);
            p->valid=false;
            // GDK takes GL coordinates. Store the bottom row first, preserving RGB bytes.
            std::vector<unsigned char> flipped(size_t(w)*h*3);
            const auto src=gdk_pixbuf_read_pixels(pixels); const int stride=gdk_pixbuf_get_rowstride(pixels);
            for(int row=0;row<h;++row) std::memcpy(flipped.data()+size_t(h-1-row)*w*3,src+size_t(row)*stride,size_t(w)*3);
            glBindTexture(GL_TEXTURE_2D,p->texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1);
            glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGB8,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,flipped.data());
            if(glGetError()!=GL_NO_ERROR) { p->failed=true; return false; }
            p->width=w; p->height=h; p->generation=generation; p->identity=pixels; p->valid=true;
        }
        rtengine::PipelineTimer timer("gl_preview_draw",width,height);
        cairo_save(cr);
        cairo_translate(cr,x,y);
        cairo_rectangle(cr,0,0,width,height); cairo_clip(cr);
        // This prototype supports scale-factor 1; other scales use Cairo.
        gdk_cairo_draw_from_gl(cr,window,p->texture,GL_TEXTURE,1,0,h-height,width,height);
        cairo_restore(cr);
        // Notify GDK before subsequent Cairo crop grids, masks and color pickers.
        gdk_window_mark_paint_from_clip(window,cr);
        return true;
    } catch(const std::bad_alloc&) { clear(); return false; }
}
bool GpuPreview::matches(GdkPixbuf *pixels)
{
    if(!p || !p->valid || !pixels || p->width!=gdk_pixbuf_get_width(pixels) || p->height!=gdk_pixbuf_get_height(pixels)) return false;
    CurrentContext restore;
    gdk_gl_context_make_current(p->context);
    glBindTexture(GL_TEXTURE_2D,p->texture);
    glPixelStorei(GL_PACK_ALIGNMENT,1); glPixelStorei(GL_PACK_ROW_LENGTH,0);
    std::vector<unsigned char> data(size_t(p->width)*p->height*3);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE,data.data());
    if(glGetError()!=GL_NO_ERROR) return false;
    const auto src=gdk_pixbuf_read_pixels(pixels); int stride=gdk_pixbuf_get_rowstride(pixels);
    for(int y=0;y<p->height;++y)
        if(std::memcmp(src+size_t(y)*stride,data.data()+size_t(p->height-1-y)*p->width*3,size_t(p->width)*3)) return false;
    return true;
}
