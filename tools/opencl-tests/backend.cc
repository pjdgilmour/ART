// SPDX-License-Identifier: GPL-3.0-or-later
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "backend.h"
#include "coefficients.h"
#include "../../rtengine/lanczos.h"
#include "kernels_source.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtengine { namespace opencl {
namespace {
void check(cl_int e) { if(e!=CL_SUCCESS) throw std::runtime_error("OpenCL error "+std::to_string(e)); }
struct Buffer {
    cl_mem mem=nullptr;
    size_t bytes=0;
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer() { clear(); }
    void clear() { if(mem) clReleaseMemObject(mem); mem=nullptr; bytes=0; }
    void reserve(cl_context ctx,size_t n) {
        if(n<=bytes) return;
        clear(); cl_int err;
        mem=clCreateBuffer(ctx,CL_MEM_READ_WRITE,n,nullptr,&err); check(err); bytes=n;
    }
};
struct Runtime {
    cl_device_id device=nullptr;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    std::map<std::string,cl_kernel> kernels;
    size_t local=256;
    cl_ulong maxAllocation=0;
    Runtime() {
        try {
            cl_uint count=0; check(clGetPlatformIDs(0,nullptr,&count));
            std::vector<cl_platform_id> platforms(count);
            check(clGetPlatformIDs(count,platforms.data(),nullptr));
            for(auto platform:platforms) {
                cl_uint n=0;
                cl_int e=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,nullptr,&n);
                if(e==CL_DEVICE_NOT_FOUND) continue;
                check(e);
                std::vector<cl_device_id> devices(n);
                check(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,n,devices.data(),nullptr));
                for(auto d:devices) {
                    cl_bool available=false,compiler=false;
                    check(clGetDeviceInfo(d,CL_DEVICE_AVAILABLE,sizeof(available),&available,nullptr));
                    check(clGetDeviceInfo(d,CL_DEVICE_COMPILER_AVAILABLE,sizeof(compiler),&compiler,nullptr));
                    if(available&&compiler) { device=d; break; }
                }
                if(device) break;
            }
            if(!device) throw std::runtime_error("No available OpenCL GPU/compiler");
            for(auto field: {CL_DEVICE_NAME,CL_DEVICE_VERSION,CL_DRIVER_VERSION,CL_DEVICE_OPENCL_C_VERSION}) {
                size_t size=0; check(clGetDeviceInfo(device,field,0,nullptr,&size));
                std::vector<char> value(size); check(clGetDeviceInfo(device,field,size,value.data(),nullptr));
                std::fprintf(stderr,"OPENCL_DEVICE field=%u value=%s\n",unsigned(field),value.data());
            }
            cl_device_fp_config fp=0;
            check(clGetDeviceInfo(device,CL_DEVICE_SINGLE_FP_CONFIG,sizeof(fp),&fp,nullptr));
            if(!(fp&CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT))
                throw std::runtime_error("Strict divide/sqrt unavailable; do not silently change precision");
            check(clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(maxAllocation),&maxAllocation,nullptr));
            cl_int e;
            context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&e); check(e);
            queue=clCreateCommandQueue(context,device,0,&e); check(e);
            const char *source=kernelsSource;
            program=clCreateProgramWithSource(context,1,&source,nullptr,&e); check(e);
            e=clBuildProgram(program,1,&device,"-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt",nullptr,nullptr);
            if(e!=CL_SUCCESS) {
                size_t size=0; clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&size);
                std::vector<char> log(size+1,0); clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,size,log.data(),nullptr);
                std::fprintf(stderr,"OPENCL_BUILD %s\n",log.data()); check(e);
            }
            for(const char *name: {"initialize","finish","divide1","divide2","divide3","multiply1","multiply2","multiply3","vertical","horizontal"}) {
                cl_kernel k=clCreateKernel(program,name,&e); check(e); kernels[name]=k;
                size_t maximum=0; check(clGetKernelWorkGroupInfo(k,device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(maximum),&maximum,nullptr));
                local=std::min(local,maximum);
            }
        } catch(...) { release(); throw; }
    }
    ~Runtime() { release(); }
    void release() {
        if(queue) clFinish(queue);
        for(auto k:kernels) clReleaseKernel(k.second);
        if(program) clReleaseProgram(program);
        if(queue) clReleaseCommandQueue(queue);
        if(context) clReleaseContext(context);
    }
    void reserve(Buffer &b,size_t bytes) {
        if(bytes>maxAllocation) throw std::runtime_error("Device allocation limit exceeded");
        b.reserve(context,bytes);
    }
    void write(Buffer &b,const void *data,size_t bytes) { check(clEnqueueWriteBuffer(queue,b.mem,CL_TRUE,0,bytes,data,0,nullptr,nullptr)); }
    void read(Buffer &b,void *data,size_t bytes) { check(clEnqueueReadBuffer(queue,b.mem,CL_TRUE,0,bytes,data,0,nullptr,nullptr)); }
    template<class T> void arg(cl_kernel k,cl_uint i,const T &v) { check(clSetKernelArg(k,i,sizeof(T),&v)); }
    void args(cl_kernel,cl_uint) {}
    template<class T,class... Rest> void args(cl_kernel k,cl_uint i,const T &v,const Rest&... rest) { arg(k,i,v); args(k,i+1,rest...); }
    template<class... Args> void run(const std::string &name,size_t n,const Args&... values) {
        auto k=kernels.at(name); args(k,0,values...);
        size_t global=(n+local-1)/local*local;
        check(clEnqueueNDRangeKernel(queue,k,1,nullptr,&global,&local,0,nullptr,nullptr));
    }
};
Runtime *runtime() {
    static std::unique_ptr<Runtime> rt=[] {
        try { return std::unique_ptr<Runtime>(new Runtime); }
        catch(const std::exception &e) { std::fprintf(stderr,"OPENCL_UNAVAILABLE %s\n",e.what()); return std::unique_ptr<Runtime>(); }
    }();
    return rt.get();
}
constexpr size_t maxPixels=size_t(64)*1024*1024;
constexpr size_t budget=size_t(1024)*1024*1024;
bool dimensions(int w,int h) { return w>0&&h>0&&size_t(w)*h<=maxPixels; }
}
bool initialize() { return runtime()!=nullptr; }
bool deconvolve(float *const *lum,float *const *blend,char *const *impulse,int w,int h,double sigma,float amount,bool multiThread)
{
    if(!dimensions(w,h)||w<8||h<8||size_t(w)*h<1024*1024||!std::isfinite(sigma)||sigma<.25||sigma>1.15||!std::isfinite(amount)||amount<=0) return false;
    auto rt=runtime(); if(!rt) return false;
    struct Pool {
        std::array<Buffer,6> buffers;
        std::unique_ptr<float[]> host;
        std::unique_ptr<unsigned char[]> mask;
        size_t capacity=0;
        bool failed=false;
    };
    // Constructed after Runtime, thus destroyed before the queue/context.
    static Pool p;
    if(p.failed) return false;
    const int n=w*h;
    try {
        if(size_t(n)>p.capacity) {
            for(auto &b:p.buffers) b.clear();
            p.capacity=0;
            p.host.reset(new float[size_t(n)*2]); p.mask.reset(new unsigned char[n]);
            for(int i=0;i<5;++i) rt->reserve(p.buffers[i],size_t(n)*sizeof(float));
            rt->reserve(p.buffers[5],n); p.capacity=n;
        }
        int invalid=0;
        #pragma omp parallel for if(multiThread) reduction(|:invalid) schedule(static)
        for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
            size_t i=size_t(y)*w+x;
            p.host[i]=lum[y][x]; p.host[n+i]=blend[y][x]; p.mask[i]=impulse[y][x];
            invalid|=!std::isfinite(lum[y][x])||!std::isfinite(blend[y][x]);
        }
        if(invalid) return false;
        auto &b=p.buffers;
        rt->write(b[0],p.host.get(),size_t(n)*sizeof(float));
        rt->write(b[1],p.host.get()+n,size_t(n)*sizeof(float));
        rt->write(b[5],p.mask.get(),n);
        const int radius=sigma<.6?1:sigma<=.84?2:3;
        auto c=coefficients(sigma,radius);
        rt->run("initialize",n,b[0].mem,b[2].mem,b[4].mem,n);
        for(int i=0;i<20;++i) {
            rt->run("divide"+std::to_string(radius),n,b[2].mem,b[0].mem,b[3].mem,w,h,c);
            rt->run("multiply"+std::to_string(radius),n,b[3].mem,b[0].mem,b[1].mem,b[5].mem,b[2].mem,b[4].mem,w,h,amount,c);
        }
        rt->run("finish",n,b[0].mem,b[1].mem,b[5].mem,b[2].mem,b[4].mem,b[3].mem,n,amount);
        rt->read(b[3],p.host.get(),size_t(n)*sizeof(float));
        #pragma omp parallel for if(multiThread) reduction(|:invalid) schedule(static)
        for(int i=0;i<n;++i) invalid|=!std::isfinite(p.host[i]);
        if(invalid) return false;
        #pragma omp parallel for if(multiThread) schedule(static)
        for(int y=0;y<h;++y) std::memcpy(lum[y],p.host.get()+size_t(y)*w,size_t(w)*sizeof(float));
        return true;
    } catch(const std::exception &e) {
        clFinish(rt->queue); p.failed=true;
        for(auto &b:p.buffers) b.clear(); p.capacity=0;
        std::fprintf(stderr,"OPENCL_DECONV_FAILED %s\n",e.what()); return false;
    }
}
struct ResizeImage::Impl {
    Runtime *rt=nullptr;
    Buffer source;
    std::array<Buffer,8> work;
    int w=0,h=0,dw=0,dh=0,support=0;
    ~Impl() { if(rt) clFinish(rt->queue); }
};
ResizeImage::ResizeImage()=default;
ResizeImage::~ResizeImage()=default;
void ResizeImage::clear() { p.reset(); }
size_t ResizeImage::deviceBytes() const {
    if(!p) return 0;
    size_t total=p->source.bytes; for(auto &b:p->work) total+=b.bytes; return total;
}
bool ResizeImage::upload(float *const *a,float *const *b,float *const *c,int w,int h)
{
    clear();
    if(!dimensions(w,h)||!a||!b||!c) return false;
    try {
        const size_t n=size_t(w)*h;
        std::unique_ptr<float[]> host(new float[n*3]);
        float *const *planes[]={a,b,c}; int finite=1;
        #pragma omp parallel for reduction(&:finite)
        for(int y=0;y<h;++y) for(int ch=0;ch<3;++ch) {
            for(int x=0;x<w;++x) finite&=std::isfinite(planes[ch][y][x]);
            std::memcpy(host.get()+ch*n+size_t(y)*w,planes[ch][y],size_t(w)*sizeof(float));
        }
        if(!finite) return false;
        auto rt=runtime(); if(!rt) return false;
        p.reset(new Impl); p->rt=rt;
        rt->reserve(p->source,n*3*sizeof(float)); rt->write(p->source,host.get(),n*3*sizeof(float));
        p->w=w; p->h=h; return true;
    } catch(const std::exception &e) { clear(); std::fprintf(stderr,"OPENCL_UPLOAD_FAILED %s\n",e.what()); return false; }
}
bool ResizeImage::resize(float *const *a,float *const *b,float *const *c,int w,int h,float scale)
{
    if(!p||!dimensions(w,h)||!a||!b||!c||!std::isfinite(scale)||scale<.05f||scale>4.f||
        w>std::ceil(p->w*scale)||h>std::ceil(p->h*scale)) return false;
    try {
        auto rt=p->rt;
        const int support=int(6.f/std::min(scale,1.f))+1;
        const size_t n=size_t(w)*h*3,mid=size_t(p->w)*h*3;
        const size_t bytes=(mid+n+size_t(w+h)*support)*sizeof(float)+size_t(w+h)*2*sizeof(int);
        if(bytes+p->source.bytes>budget) return false;
        std::vector<float> wh(size_t(w)*support,0),wv(size_t(h)*support,0);
        std::vector<int> fh(w),lh(w),fv(h),lv(h);
        lanczosWeights(p->w,w,scale,support,wh.data(),fh.data(),lh.data());
        lanczosWeights(p->h,h,scale,support,wv.data(),fv.data(),lv.data());
        for(int i=0;i<w;++i) if(fh[i]>=lh[i]) return false;
        for(int i=0;i<h;++i) if(fv[i]>=lv[i]) return false;
        auto &d=p->work;
        if(w!=p->dw||h!=p->dh||support!=p->support) {
            for(auto &b:d) b.clear();
            p->dw=w; p->dh=h; p->support=support;
        }
        const size_t sizes[]={mid*sizeof(float),n*sizeof(float),wh.size()*sizeof(float),wv.size()*sizeof(float),size_t(w)*sizeof(int),size_t(w)*sizeof(int),size_t(h)*sizeof(int),size_t(h)*sizeof(int)};
        for(int i=0;i<8;++i) rt->reserve(d[i],sizes[i]);
        rt->write(d[2],wh.data(),sizes[2]); rt->write(d[3],wv.data(),sizes[3]);
        rt->write(d[4],fh.data(),sizes[4]); rt->write(d[5],lh.data(),sizes[5]);
        rt->write(d[6],fv.data(),sizes[6]); rt->write(d[7],lv.data(),sizes[7]);
        std::unique_ptr<float[]> host(new float[n]);
        rt->run("vertical",mid,p->source.mem,d[0].mem,p->w,p->h,h,d[3].mem,d[6].mem,d[7].mem,support);
        rt->run("horizontal",n,d[0].mem,d[1].mem,p->w,w,h,d[2].mem,d[4].mem,d[5].mem,support);
        rt->read(d[1],host.get(),n*sizeof(float));
        int finite=1;
        #pragma omp parallel for reduction(&:finite)
        for(size_t i=0;i<n;++i) finite&=std::isfinite(host[i]);
        if(!finite) { clear(); return false; }
        float *const *planes[]={a,b,c};
        #pragma omp parallel for
        for(int y=0;y<h;++y) for(int ch=0;ch<3;++ch)
            std::memcpy(planes[ch][y],host.get()+size_t(ch)*w*h+size_t(y)*w,size_t(w)*sizeof(float));
        return true;
    } catch(const std::exception &e) { clear(); std::fprintf(stderr,"OPENCL_RESIZE_FAILED %s\n",e.what()); return false; }
}
} }
