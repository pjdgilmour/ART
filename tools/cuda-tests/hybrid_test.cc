// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../rtengine/deconvolution.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
struct Plane {
    std::vector<float> values;
    std::vector<float *> rows;
    Plane(int w,int h): values(size_t(w)*h),rows(h) {
        for(int y=0;y<h;++y) rows[y]=values.data()+size_t(y)*w;
    }
};
int main(int argc,char **argv) {
    const int w=argc>1?std::atoi(argv[1]):4097, h=argc>2?std::atoi(argv[2]):2051;
    if(w<8 || h<8) return 2;
    Plane a(w,h),b(w,h),referenceA(w,h),referenceB(w,h),blend(w,h);
    std::vector<char> impulse(size_t(w)*h);
    std::vector<char*> rows(h);
    for(int y=0;y<h;++y) rows[y]=impulse.data()+size_t(y)*w;
    for(size_t i=0;i<a.values.size();++i) {
        a.values[i]=b.values[i]=referenceA.values[i]=referenceB.values[i]=float(i%65537)-100.f;
        blend.values[i]=float(i%13)/12.f;
        impulse[i]=i%19==0;
    }
    using Clock=std::chrono::steady_clock;
    auto start=Clock::now();
    rtengine::deconvsharpening(referenceA.rows.data(),blend.rows.data(),rows.data(),w,h,1.27,.7f,true,true);
    rtengine::deconvsharpening(referenceB.rows.data(),blend.rows.data(),rows.data(),w,h,1.31,.7f,true,true);
    const double cpu=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    start=Clock::now();
    rtengine::deconvsharpeningPair(a.rows.data(),b.rows.data(),blend.rows.data(),rows.data(),w,h,1.27,1.31,.7f,true);
    const double hybrid=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    size_t different=0;
    double maximum=0;
    for(size_t i=0;i<a.values.size();++i) {
        if(!std::isfinite(a.values[i]) || !std::isfinite(b.values[i])) return 1;
        different+=(a.values[i]!=referenceA.values[i])+(b.values[i]!=referenceB.values[i]);
        maximum=std::max(maximum,std::max(std::abs(double(a.values[i])-referenceA.values[i]),std::abs(double(b.values[i])-referenceB.values[i])));
    }
    std::printf("%dx%d CPU_pair=%.3fms hybrid_pair=%.3fms different=%zu max_abs=%.9g\n",w,h,cpu,hybrid,different,maximum);
    return different?1:0;
}
