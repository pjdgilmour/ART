// SPDX-License-Identifier: GPL-3.0-or-later
// Same process, same input, rotating backend order; validation outside timing.
#include "backend.h"
#include "../../rtengine/lanczos.h"
#include "../../rtengine/deconvolution.h"
#ifdef ART_COMPARE_CUDA
#include "../../rtengine/cuda/deconvolution.h"
#include "../../rtengine/cuda/resize.h"
#endif
#include <tiffio.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
template<class T> struct Plane {
    int w,h,stride;
    std::vector<T> data;
    std::vector<T*> row;
    Plane(int width,int height):w(width),h(height),stride(w+7),data(size_t(stride)*h),row(h) {
        for(int y=0;y<h;++y) row[y]=data.data()+size_t(y)*stride;
    }
    void copy(const Plane &other) { for(int y=0;y<h;++y) std::memcpy(row[y],other.row[y],size_t(w)*sizeof(T)); }
};
struct Image {
    Plane<float> a,b,c;
    Image(int w,int h):a(w,h),b(w,h),c(w,h) {}
};
struct Difference {
    double maximum=0;
    size_t different=0;
    void add(const Plane<float> &a,const Plane<float> &b) {
        for(int y=0;y<a.h;++y) for(int x=0;x<a.w;++x) {
            double av=a.row[y][x],bv=b.row[y][x],d=std::abs(av-bv);
            require(std::isfinite(av)&&std::isfinite(bv),"Non-finite output");
            require(d<=.05+.000005*std::abs(av),"Numerical mismatch");
            maximum=std::max(maximum,d); different+=av!=bv;
        }
    }
    void add(const Image &a,const Image &b) { add(a.a,b.a); add(a.b,b.b); add(a.c,b.c); }
};
void log(const char *op,const std::string &backend,int run,int w,int h,double sigma,double ms,const Difference &d,size_t bytes=0) {
    std::printf("{\"operation\":\"%s\",\"backend\":\"%s\",\"run\":%d,\"width\":%d,\"height\":%d,\"sigma\":%.10g,\"ms\":%.6f,\"max_abs\":%.9g,\"different\":%zu,\"device_bytes\":%zu}\n",op,backend.c_str(),run,w,h,sigma,ms,d.maximum,d.different,bytes);
    std::fflush(stdout);
}
int main(int argc,char **argv) {
    if(argc!=2) { std::fprintf(stderr,"usage: gpu_benchmark RGB16_TIFF\n"); return 2; }
    try {
        TIFF *t=TIFFOpen(argv[1],"r"); require(t,"Open TIFF failed");
        uint32_t w=0,h=0; uint16_t bits=0,channels=0,planar=0,format=SAMPLEFORMAT_UINT;
        TIFFGetField(t,TIFFTAG_IMAGEWIDTH,&w); TIFFGetField(t,TIFFTAG_IMAGELENGTH,&h);
        TIFFGetField(t,TIFFTAG_BITSPERSAMPLE,&bits); TIFFGetField(t,TIFFTAG_SAMPLESPERPIXEL,&channels);
        TIFFGetField(t,TIFFTAG_PLANARCONFIG,&planar); TIFFGetFieldDefaulted(t,TIFFTAG_SAMPLEFORMAT,&format);
        require(bits==16&&channels==3&&planar==PLANARCONFIG_CONTIG&&format==SAMPLEFORMAT_UINT,"Requires RGB16 TIFF");
        require(w>=1024&&h>=1024&&size_t(w)*h<=64*1024*1024,"Unsupported dimensions");
        Image src(w,h);
        std::vector<uint16_t> scan(size_t(TIFFScanlineSize(t))/2);
        for(unsigned y=0;y<h;++y) {
            require(TIFFReadScanline(t,scan.data(),y)>=0,"Read TIFF failed");
            for(unsigned x=0;x<w;++x) {
                src.a.row[y][x]=scan[x*3]; src.b.row[y][x]=scan[x*3+1]; src.c.row[y][x]=scan[x*3+2];
            }
        }
        TIFFClose(t);
        std::vector<std::string> backends={"cpu","opencl"};
#ifdef ART_COMPARE_CUDA
        backends.push_back("cuda");
#endif
        {
            Plane<float> reference(w,h),work(w,h),blend(w,h);
            Plane<char> impulse(w,h);
            for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
                blend.row[y][x]=float((x+y)%11)/10.f; impulse.row[y][x]=(x+y)%17==0;
            }
            for(double sigma: {.348989695,.93307381868362427}) {
                reference.copy(src.b);
                rtengine::deconvsharpening(reference.row.data(),blend.row.data(),impulse.row.data(),w,h,sigma,.7f,true);
                for(int run=0;run<7;++run) for(size_t j=0;j<backends.size();++j) {
                    const auto &name=backends[(j+run)%backends.size()]; work.copy(src.b);
                    auto start=Clock::now(); bool used=true;
                    if(name=="cpu") rtengine::deconvsharpening(work.row.data(),blend.row.data(),impulse.row.data(),w,h,sigma,.7f,true);
                    else if(name=="opencl") used=rtengine::opencl::deconvolve(work.row.data(),blend.row.data(),impulse.row.data(),w,h,sigma,.7f);
#ifdef ART_COMPARE_CUDA
                    else used=rtengine::cuda::deconvolve(work.row.data(),blend.row.data(),impulse.row.data(),w,h,sigma,.7f);
#endif
                    double time=elapsed(start); require(used,"GPU backend was not executed");
                    Difference d; d.add(reference,work); log("deconvolution",name,run,w,h,sigma,time,d);
                }
            }
        }
        for(float scale: {2560.f/6732.f,.25f}) {
            const int dw=int(w*scale),dh=int(h*scale);
            Image reference(dw,dh),work(dw,dh);
            auto cpu=[&](Image &dst) { rtengine::lanczosCpu(src.a.row.data(),src.b.row.data(),src.c.row.data(),w,h,dst.a.row.data(),dst.b.row.data(),dst.c.row.data(),dw,dh,scale); };
            cpu(reference);
            rtengine::opencl::ResizeImage clCache;
#ifdef ART_COMPARE_CUDA
            rtengine::cuda::ResizeImage cudaCache;
#endif
            for(int run=0;run<7;++run) for(size_t j=0;j<backends.size();++j) {
                const auto &name=backends[(j+run)%backends.size()];
                if(name=="cpu") {
                    auto start=Clock::now(); cpu(work); double time=elapsed(start);
                    Difference d; d.add(reference,work); log("resize",name,run,dw,dh,0,time,d); continue;
                }
                auto measure=[&](auto &cache) {
                    auto start=Clock::now();
                    require(cache.upload(src.a.row.data(),src.b.row.data(),src.c.row.data(),w,h),"Upload failed");
                    double upload=elapsed(start);
                    require(cache.resize(work.a.row.data(),work.b.row.data(),work.c.row.data(),dw,dh,scale),"Resize failed");
                    double time=elapsed(start);
                    Difference d; d.add(reference,work);
                    log("resize_new_source",name,run,dw,dh,0,time,d,cache.deviceBytes());
                    log("resize_upload",name,run,dw,dh,0,upload,Difference(),cache.deviceBytes());
                    start=Clock::now();
                    require(cache.resize(work.a.row.data(),work.b.row.data(),work.c.row.data(),dw,dh,scale),"Resident resize failed");
                    time=elapsed(start); Difference resident; resident.add(reference,work);
                    log("resize_resident",name,run,dw,dh,0,time,resident,cache.deviceBytes());
                };
                if(name=="opencl") measure(clCache);
#ifdef ART_COMPARE_CUDA
                else measure(cudaCache);
#endif
            }
        }
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
