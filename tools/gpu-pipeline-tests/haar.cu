// SPDX-License-Identifier: GPL-3.0-or-later
// Arithmetic/boundaries from ART cplx_wavelet_level.h (Popov/Martinec/Weyrich).
#include "haar.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace experiment {
namespace {
void check(cudaError_t error) {
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void haar(const float *src,float *low,float *bands,int w,int h,int skip) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    const int x=i%w,y=i/w;
    const int xx=x<w-skip?x+skip:x-skip;
    const int yy=y<h-skip?y+skip:y-skip;
    const float a=.25f*(src[i]+src[yy*w+x]);
    const float b=.25f*(src[y*w+xx]+src[yy*w+xx]);
    const float c=.25f*(src[i]-src[yy*w+x]);
    const float d=.25f*(src[y*w+xx]-src[yy*w+xx]);
    low[i]=a+b;
    bands[i]=a-b;
    bands[w*h+i]=c+d;
    bands[2*w*h+i]=c-d;
}
}
Haar::Haar(int w,int h,int levels):w_(w),h_(h),levels_(levels) {
    if(levels<1||levels>8||w<2*(1<<(levels-1))||h<2*(1<<(levels-1))||size_t(w)*h>32*1024*1024) throw std::runtime_error("unsupported pyramid dimensions");
    check(cudaMalloc(&device_,bytes()));
}
Haar::~Haar() { if(device_) cudaFree(device_); }
size_t Haar::bytes() const { return size_t(w_)*h_*(3*levels_+2)*sizeof(float); }
void Haar::upload(const float *input) { check(cudaMemcpy(device_,input,size_t(w_)*h_*sizeof(float),cudaMemcpyHostToDevice)); }
void Haar::run() {
    const size_t n=size_t(w_)*h_;
    for(int l=0;l<levels_;++l) {
        haar<<<(n+255)/256,256>>>(device_+(l%2)*n,device_+((l+1)%2)*n,device_+(2+3*l)*n,w_,h_,1<<l);
    }
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}
void Haar::download(float *output) {
    const size_t n=size_t(w_)*h_;
    check(cudaMemcpy(output,device_+2*n,3*levels_*n*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(output+3*levels_*n,device_+(levels_%2)*n,n*sizeof(float),cudaMemcpyDeviceToHost));
}
}
