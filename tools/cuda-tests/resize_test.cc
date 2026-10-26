// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../rtengine/lanczos.h"
#ifdef ART_TEST_OPENCL
#include "../opencl-tests/backend.h"
namespace testgpu = rtengine::opencl;
#else
#include "../../rtengine/cuda/resize.h"
namespace testgpu = rtengine::cuda;
#endif
#include <tiffio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
void check(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
struct Image {
    int w,h,stride;
    std::vector<float> data[3];
    std::vector<float*> row[3];
    Image(int width,int height):w(width),h(height),stride(w+7) {
        for (int c=0;c<3;++c) {
            data[c].resize(size_t(stride)*h,-123.f); row[c].resize(h);
            for (int y=0;y<h;++y) row[c][y]=data[c].data()+size_t(y)*stride;
        }

    }
    float **a() { return row[0].data(); }
    float **b() { return row[1].data(); }
    float **c() { return row[2].data(); }
};
void compare(const Image &a,const Image &b) {
    size_t different=0; double max=0;
    for(int c=0;c<3;++c) for(int y=0;y<a.h;++y) {
        for(int x=0;x<a.w;++x) {
            float av=a.row[c][y][x],bv=b.row[c][y][x];
            check(std::isfinite(av)&&std::isfinite(bv),"non-finite output");
            double err=std::abs(double(av)-bv); max=std::max(max,err); different+=av!=bv;
            check(err<=.05+std::abs(av)*.000005,"numerical mismatch");
        }
        for(int x=a.w;x<a.stride;++x) check(b.row[c][y][x]==-123.f,"padding overwritten");
    }
    std::printf(" max_abs=%.9g different=%zu\n",max,different);
}
int main(int argc,char **argv) {
    try {
        bool noDevice=argc>1 && std::string(argv[1])=="--no-device";
        TIFF *tif=nullptr; uint32_t w=1027,h=769;
        if(argc>1 && !noDevice) {
            tif=TIFFOpen(argv[1],"r"); check(tif,"open TIFF");
            uint16_t bits=0,channels=0,planar=0,format=SAMPLEFORMAT_UINT;
            TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&w); TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&h);
            TIFFGetField(tif,TIFFTAG_BITSPERSAMPLE,&bits); TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&channels);
            TIFFGetField(tif,TIFFTAG_PLANARCONFIG,&planar); TIFFGetFieldDefaulted(tif,TIFFTAG_SAMPLEFORMAT,&format);
            check(bits==16&&channels==3&&planar==PLANARCONFIG_CONTIG&&format==SAMPLEFORMAT_UINT,"requires RGB16 TIFF");
        }
        Image source(w,h);
        if(tif) {
            std::vector<uint16_t> scan(size_t(TIFFScanlineSize(tif))/2);
            for(unsigned y=0;y<h;++y) {
                check(TIFFReadScanline(tif,scan.data(),y)>=0,"TIFF read");
                for(unsigned x=0;x<w;++x) for(int c=0;c<3;++c) source.row[c][y][x]=scan[3*x+c];
            }
            TIFFClose(tif);
        } else {
            unsigned rnd=42;
            for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) for(int c=0;c<3;++c) {
                rnd=rnd*1664525+1013904223;
                source.row[c][y][x]=c==0?2000.f:c==1?float(rnd>>16)*1.5f-500.f:(x<w/2?-500.f:60000.f);
            }
        }
        testgpu::ResizeImage cache;
        if(noDevice) {
            check(!cache.upload(source.a(),source.b(),source.c(),w,h),"device unexpectedly available");
            Image dst(100,100);
            check(!cache.resize(dst.a(),dst.b(),dst.c(),100,100,.1f),"empty cache accepted");
            check(dst.data[0][0]==-123.f&&cache.deviceBytes()==0,"failed cache not empty");
            std::puts("PASS no-device fallback"); return 0;
        }
        for(float scale: {2560.f/6732.f,.25f,1.f,1.3f}) {
            if(w>2000 && scale>=1) continue; // peak-memory budget, tested on small images instead
            int dw=int(w*scale),dh=int(h*scale);
            Image cpu(dw,dh),gpu(dw,dh);
            for(int run=0;run<5;++run) {
                auto start=Clock::now();
                rtengine::lanczosCpu(source.a(),source.b(),source.c(),w,h,cpu.a(),cpu.b(),cpu.c(),dw,dh,scale);
                double cpuMs=ms(start);
                start=Clock::now();
                check(cache.upload(source.a(),source.b(),source.c(),w,h),"GPU upload failed");
                double uploadMs=ms(start);
                check(cache.resize(gpu.a(),gpu.b(),gpu.c(),dw,dh,scale),"GPU resize failed");
                double coldMs=ms(start);
                std::printf("PASS %ux%u -> %dx%d run=%d cpu=%.3fms upload=%.3fms cold_total=%.3fms bytes=%zu",w,h,dw,dh,run,cpuMs,uploadMs,coldMs,cache.deviceBytes());
                compare(cpu,gpu);
                start=Clock::now();
                check(cache.resize(gpu.a(),gpu.b(),gpu.c(),dw,dh,scale),"resident resize failed");
                std::printf("PASS resident=%.3fms",ms(start)); compare(cpu,gpu);
                std::fflush(stdout);
            }
        }
        // Reuse one source allocation across changing destination dimensions.
        for(int dw: {7,63,17}) {
            Image cpu(dw,13),gpu(dw,13);
            rtengine::lanczosCpu(source.a(),source.b(),source.c(),w,h,cpu.a(),cpu.b(),cpu.c(),dw,13,.5f);
            check(cache.resize(gpu.a(),gpu.b(),gpu.c(),dw,13,.5f),"changing destination size failed");
            std::printf("PASS destination change"); compare(cpu,gpu);
        }
        // Replace the source in-place: explicit upload must invalidate old pixels.
        source.row[0][0][0]+=10000.f;
        check(cache.upload(source.a(),source.b(),source.c(),w,h),"replace upload failed");
        Image cpu(7,5),gpu(7,5);
        rtengine::lanczosCpu(source.a(),source.b(),source.c(),w,h,cpu.a(),cpu.b(),cpu.c(),7,5,1.f);
        check(cache.resize(gpu.a(),gpu.b(),gpu.c(),7,5,1.f),"replacement resize failed");
        std::printf("PASS replacement"); compare(cpu,gpu);
        auto original=gpu.data[0];
        check(!cache.resize(gpu.a(),gpu.b(),gpu.c(),7,5,.001f),"unsupported scale accepted");
        check(gpu.data[0]==original,"fallback touched output");
        check(!cache.resize(gpu.a(),gpu.b(),gpu.c(),100000,100000,1.f),"oversized output accepted");
        check(gpu.data[0]==original,"oversized fallback touched output");
        source.row[0][0][0]=std::numeric_limits<float>::infinity();
        check(!cache.upload(source.a(),source.b(),source.c(),w,h),"non-finite accepted");
        check(cache.deviceBytes()==0,"stale cache retained");
        check(!cache.resize(gpu.a(),gpu.b(),gpu.c(),7,5,1.f),"invalid cache accepted");
        cache.clear(); check(cache.deviceBytes()==0,"cache release failed");
        check(!cache.upload(source.a(),source.b(),source.c(),100000,100000),"oversized source accepted");
        std::puts("PASS replacement, non-finite/unsupported fallback, release");
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
