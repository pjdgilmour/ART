// SPDX-License-Identifier: GPL-3.0-or-later
#include "rawconvert.h"
#include "cpupolicy.h"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ART_TARGET_AVX2
#endif
namespace rtengine {
namespace {
void scalar(const uint16_t *src,float *dst,int n,int a,int b) {
    for(int x=0;x<n;++x) dst[x]=src[4*x+((x&1)?b:a)];
}
#ifdef ART_TARGET_AVX2
__attribute__((target("avx2"))) void avx2(const uint16_t *src,float *dst,int n,int a,int b) {
    const __m256i indices=_mm256_setr_epi32(a/2,2+b/2,4+a/2,6+b/2,8+a/2,10+b/2,12+a/2,14+b/2);
    const __m256i shift=_mm256_setr_epi32((a&1)*16,(b&1)*16,(a&1)*16,(b&1)*16,(a&1)*16,(b&1)*16,(a&1)*16,(b&1)*16);
    const __m256i mask=_mm256_set1_epi32(65535);
    int x=0;
    for(;x+8<=n;x+=8) {
        const auto v=_mm256_i32gather_epi32(reinterpret_cast<const int*>(src+4*x),indices,4);
        _mm256_storeu_ps(dst+x,_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(v,shift),mask)));
    }
    scalar(src+4*x,dst+x,n-x,a,b);
}
#endif
}
void unpackBayerRow(const uint16_t *src,float *dst,int n,int a,int b,bool simd) {
#ifdef ART_TARGET_AVX2
    if(simd && cpuTopology().avx2) {avx2(src,dst,n,a,b);return;}
#endif
    scalar(src,dst,n,a,b);
}
}
