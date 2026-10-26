// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../rtengine/rawconvert.h"
#include <vector>
#include <chrono>
#include <cstdio>
int main() {
 const int w=6732,h=4492;std::vector<uint16_t> input(size_t(w)*h*4,32000);std::vector<float> output(size_t(w)*h);
 for(int run=0;run<6;++run) {
  bool simd=run%2;auto start=std::chrono::steady_clock::now();
#pragma omp parallel for
  for(int y=0;y<h;++y) rtengine::unpackBayerRow(input.data()+size_t(y)*w*4,output.data()+size_t(y)*w,w,y%2,1+y%2,simd);
  std::printf("%s %.3f ms %.0f\n",simd?"avx2":"compiler",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),output.back());
 }
}
