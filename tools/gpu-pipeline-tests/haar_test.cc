// SPDX-License-Identifier: GPL-3.0-or-later
#include "haar.h"
#include "../../rtengine/cplx_wavelet_level.h"
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cuda_runtime.h>
int main(int argc,char **argv) {
 try {
    const int w=argc>1?std::atoi(argv[1]):513,h=argc>2?std::atoi(argv[2]):515,levels=5;
    const size_t n=size_t(w)*h;
    std::vector<float> input(n),low(n),scratch(n),out(n*(3*levels+1));
    for(size_t i=0;i<n;++i) input[i]=float((i*1664525u+1013904223u)%100003)-1000.f;
    using Clock=std::chrono::steady_clock;
    const auto elapsed=[](Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    experiment::Haar gpu(w,h,levels);
    const bool pinned=argc>3;
    if(pinned) {
        auto start=Clock::now();
        if(cudaHostRegister(input.data(),n*sizeof(float),cudaHostRegisterDefault)!=cudaSuccess ||
           cudaHostRegister(out.data(),out.size()*sizeof(float),cudaHostRegisterDefault)!=cudaSuccess) return 1;
        std::printf("host_registration_ms=%.3f\n",elapsed(start));
    }
    for(int run=0;run<4;++run) {
        std::vector<std::unique_ptr<rtengine::wavelet_level<float>>> cpu;
        auto start=Clock::now();
        for(int l=0;l<levels;++l) {
            float *src=l==0?input.data():(l%2?low.data():scratch.data());
            float *dst=l%2?scratch.data():low.data();
            cpu.emplace_back(new rtengine::wavelet_level<float>(src,dst,l,0,w,h,nullptr,nullptr,6,2,1,omp_get_max_threads()));
        }
        double cpu_ms=elapsed(start);
        start=Clock::now();gpu.upload(input.data());double upload=elapsed(start);
        start=Clock::now();gpu.run();double kernel=elapsed(start);
        start=Clock::now();gpu.download(out.data());double download=elapsed(start);
        size_t different=0; double maximum=0;
        for(int l=0;l<levels;++l) for(int b=1;b<4;++b) for(size_t i=0;i<n;++i) {
            float a=cpu[l]->subbands()[b][i],v=out[(3*l+b-1)*n+i];
            if(!std::isfinite(v)) return 1;
            different+=a!=v;maximum=std::max(maximum,std::abs(double(a)-v));
        }
        for(size_t i=0;i<n;++i) {float a=low[i],b=out[3*levels*n+i];different+=a!=b;maximum=std::max(maximum,std::abs(double(a)-b));}
        std::printf("run=%d %dx%d levels=%d cpu_ms=%.3f upload_ms=%.3f kernel_ms=%.3f download_ms=%.3f total_ms=%.3f vram_bytes=%zu different=%zu max_abs=%.9g\n",run,w,h,levels,cpu_ms,upload,kernel,download,upload+kernel+download,gpu.bytes(),different,maximum);
        if(different) return 1;
    }
    if(pinned) {cudaHostUnregister(out.data());cudaHostUnregister(input.data());}
    return 0;
 } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
