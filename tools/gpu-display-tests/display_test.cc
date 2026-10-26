// SPDX-License-Identifier: GPL-3.0-or-later
// Actual GTK/GDK window benchmark; never interpret callback time alone as FPS.
#include "gpupreview.h"
#include <gtk/gtk.h>
#include <epoxy/gl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock=std::chrono::steady_clock;
struct Test {
    GtkWidget *area=nullptr;
    GdkPixbuf *pixels=nullptr,*baseline=nullptr;
    GpuPreview cache;
    int phase=0,frame=0;
    uint64_t generation=1;
    Clock::time_point start;
    std::vector<double> times[4];
    bool failed=false,painted=false;
    bool expectNoGL=std::getenv("ART_TEST_EXPECT_NO_GL")!=nullptr;
};
gboolean draw(GtkWidget *area,cairo_t *cr,gpointer arg) {
    auto &t=*static_cast<Test*>(arg); t.start=Clock::now(); t.painted=true;
    cairo_set_source_rgb(cr,.25,.25,.25); cairo_paint(cr);
    const int w=std::min(1100,gdk_pixbuf_get_width(t.pixels)),h=std::min(700,gdk_pixbuf_get_height(t.pixels));
    // Exercise a non-zero origin and a non-trivial clip, as in CropWindow.
    cairo_rectangle(cr,17,23,w-21,h-25); cairo_clip(cr);
    bool drawn=false;
    if(t.phase==1 || t.phase==2) {
        if(t.phase==1) ++t.generation;
        drawn=t.cache.draw(cr,gtk_widget_get_window(area),t.pixels,t.generation,7,11,w,h);
        if(drawn==t.expectNoGL) t.failed=true;
    }
    if(!drawn) {
        gdk_cairo_set_source_pixbuf(cr,t.pixels,7,11);
        cairo_rectangle(cr,7,11,w,h); cairo_fill(cr);
    }
    // Cairo overlays must still work on top of the GL image.
    cairo_set_source_rgb(cr,1,0,0); cairo_rectangle(cr,90,90,75,37); cairo_fill(cr);
    return TRUE;
}
gboolean redraw(gpointer arg) {
    gtk_widget_queue_draw(static_cast<Test*>(arg)->area); return G_SOURCE_REMOVE;
}
void afterPaint(GdkFrameClock*,gpointer arg) {
    auto &t=*static_cast<Test*>(arg);
    if(!t.painted) return;
    t.painted=false;
    auto current=gdk_gl_context_get_current();
    if(current) glFinish();
    gdk_display_sync(gtk_widget_get_display(t.area));
    const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-t.start).count();
    if(t.frame==0) std::printf("FIRST_FRAME phase=%d ms=%.3f\n",t.phase,elapsed);
    if(t.frame>=5) t.times[t.phase].push_back(elapsed);
    if(t.frame==6) {
        if((t.phase==1 || t.phase==2) && !t.expectNoGL) {
            bool matches=t.cache.matches(t.pixels);
            std::printf("texture phase=%d exact=%d vram_budget_bytes=%zu\n",t.phase,matches,t.cache.deviceBytes());
            t.failed|=!matches;
        }
        auto capture=gdk_pixbuf_get_from_window(gtk_widget_get_window(t.area),0,0,1150,750);
        if(capture && std::getenv("ART_TEST_CAPTURE")) {
            char path[128]; std::snprintf(path,sizeof(path),"/tmp/art-display-phase-%d.png",t.phase);
            gdk_pixbuf_save(capture,path,"png",nullptr,nullptr);
        }
        if(!capture) t.failed=true;
        else if(!t.baseline) t.baseline=capture;
        else {
            size_t diff=0; int max=0;
            const auto a=gdk_pixbuf_read_pixels(t.baseline),b=gdk_pixbuf_read_pixels(capture);
            const int sa=gdk_pixbuf_get_rowstride(t.baseline),sb=gdk_pixbuf_get_rowstride(capture);
            for(int y=0;y<750;++y) for(int x=0;x<1150*3;++x) {
                int d=std::abs(int(a[size_t(y)*sa+x])-b[size_t(y)*sb+x]); diff+=d!=0; max=std::max(max,d);
            }
            std::printf("window phase=%d different=%zu max_lsb=%d\n",t.phase,diff,max);
            t.failed|=diff!=0; g_object_unref(capture);
        }
    }
    if(++t.frame==35) {
        auto &v=t.times[t.phase]; std::sort(v.begin(),v.end());
        const char *names[]={"cairo","gl_upload_each_frame","gl_resident","cairo_repeat"};
        if(t.expectNoGL) names[1]=names[2]="cairo_fallback";
        std::printf("TIMING mode=%s samples=%zu median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",names[t.phase],v.size(),v[v.size()/2],v.front(),v.back());
        std::fflush(stdout); t.frame=0;
        if(++t.phase==4) { gtk_main_quit(); return; }
    }
    g_idle_add(redraw,&t);
}
int main(int argc,char **argv) {
    if(argc<2) { std::fprintf(stderr,"usage: display_test IMAGE [full]\n"); return 2; }
    gtk_init(&argc,&argv);
    Test t; GError *error=nullptr;
    GdkPixbuf *source=gdk_pixbuf_new_from_file(argv[1],&error);
    if(!source) { std::fprintf(stderr,"%s\n",error->message); g_error_free(error); return 1; }
    if(gdk_pixbuf_get_has_alpha(source)) {
        const int w=gdk_pixbuf_get_width(source),h=gdk_pixbuf_get_height(source);
        auto rgb=gdk_pixbuf_new(GDK_COLORSPACE_RGB,FALSE,8,w,h);
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            auto s=gdk_pixbuf_read_pixels(source)+size_t(y)*gdk_pixbuf_get_rowstride(source)+x*4;
            auto d=gdk_pixbuf_get_pixels(rgb)+size_t(y)*gdk_pixbuf_get_rowstride(rgb)+x*3;
            if(s[3]!=255) { std::fprintf(stderr,"input must be opaque\n"); return 2; }
            for(int c=0;c<3;++c) d[c]=s[c];
        }
        g_object_unref(source); source=rgb;
    }
    if(argc>2) t.pixels=source;
    else {
        double scale=1100./gdk_pixbuf_get_width(source);
        t.pixels=gdk_pixbuf_scale_simple(source,1100,int(gdk_pixbuf_get_height(source)*scale),GDK_INTERP_BILINEAR);
        g_object_unref(source);
    }
    std::printf("source=%dx%d channels=%d\n",gdk_pixbuf_get_width(t.pixels),gdk_pixbuf_get_height(t.pixels),gdk_pixbuf_get_n_channels(t.pixels));
    GtkWidget *window=gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window),"ART GPU presentation test (temporary)");
    gtk_window_set_default_size(GTK_WINDOW(window),1150,750);
    t.area=gtk_drawing_area_new(); gtk_container_add(GTK_CONTAINER(window),t.area);
    g_signal_connect(t.area,"draw",G_CALLBACK(draw),&t);
    g_signal_connect(window,"destroy",G_CALLBACK(+[](GtkWidget*,gpointer) { if(gtk_main_level()) gtk_main_quit(); }),nullptr);
    gtk_widget_show_all(window);
    g_signal_connect_after(gtk_widget_get_frame_clock(t.area),"after-paint",G_CALLBACK(afterPaint),&t);
    gtk_main();
    if(t.phase==4 && !t.expectNoGL) {
        // Same storage, new contents: generation must refresh the resident copy.
        auto bytes=gdk_pixbuf_get_pixels(t.pixels);
        bytes[0]^=255;
        t.failed|=t.cache.matches(t.pixels);
        auto surface=cairo_image_surface_create(CAIRO_FORMAT_RGB24,64,64);
        auto cr=cairo_create(surface);
        t.failed|=!t.cache.draw(cr,gtk_widget_get_window(t.area),t.pixels,++t.generation,0,0,64,64);
        t.failed|=!t.cache.matches(t.pixels);
        cairo_destroy(cr); cairo_surface_destroy(surface);
        std::puts("CHECK same-buffer generation invalidation");
    }
    t.cache.clear();
    t.failed|=t.cache.deviceBytes()!=0;
    if(t.baseline) g_object_unref(t.baseline);
    g_object_unref(t.pixels); gtk_widget_destroy(window);
    return t.failed || t.phase!=4 ? 1 : 0;
}
