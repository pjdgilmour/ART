// SPDX-License-Identifier: GPL-3.0-or-later
// Port of gauss.cc Young/van Vliet recursion, including mixed precision tails.
// Original: Copyright (c) 2004-2010 Gabor Horvath.
namespace recursive {
template<class T> void calculateFactors( const T sigma, T &b1, T &b2, T &b3, T &B, T M[3][3])
{
    // coefficient calculation
    T q;

    if (sigma < 2.5) {
        q = 3.97156 - 4.14554 * sqrt (1.0 - 0.26891 * sigma);
    } else {
        q = 0.98711 * sigma - 0.96330;
    }

    T b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    b2 = -1.4281 * q * q - 1.26661 * q * q * q;
    b3 = 0.422205 * q * q * q;
    B = 1.0 - (b1 + b2 + b3) / b0;

    b1 /= b0;
    b2 /= b0;
    b3 /= b0;

    // From: Bill Triggs, Michael Sdika: Boundary Conditions for Young-van Vliet Recursive Filtering
    M[0][0] = -b3 * b1 + 1.0 - b3 * b3 - b2;
    M[0][1] = (b3 + b1) * (b2 + b3 * b1);
    M[0][2] = b3 * (b1 + b3 * b2);
    M[1][0] = b1 + b3 * b2;
    M[1][1] = -(b2 - 1.0) * (b2 + b3 * b1);
    M[1][2] = -(b3 * b1 + b3 * b3 + b2 - 1.0) * b3;
    M[2][0] = b3 * b1 + b2 + b1 * b1 - b2 * b2;
    M[2][1] = b1 * b2 + b3 * b2 * b2 - b1 * b3 * b3 - b3 * b3 * b3 - b3 * b2 + b3;
    M[2][2] = b3 * (b1 + b3 * b2);

}

template<class T> struct Factors { T b1,b2,b3,B,M[3][3]; };
struct Coefficients { Factors<float> f; Factors<double> d; };
inline Coefficients factors(double sigma) {
    Coefficients c;
    calculateFactors<double>(float(sigma),c.d.b1,c.d.b2,c.d.b3,c.d.B,c.d.M);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        c.d.M[i][j] *= 1.0+c.d.b2+(c.d.b1-c.d.b3)*c.d.b3;
        c.d.M[i][j] /= (1.0+c.d.b1-c.d.b2+c.d.b3)*(1.0-c.d.b1-c.d.b2-c.d.b3);
        c.f.M[i][j]=c.d.M[i][j];
    }
    c.f.b1=c.d.b1;c.f.b2=c.d.b2;c.f.b3=c.d.b3;c.f.B=c.d.B;
    return c;
}
// Coalesced transpose makes the serial horizontal recurrence a column pass.
__global__ void transpose(const float *src,float *dst,int w,int h) {
    __shared__ float tile[32][33];
    int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(x<w&&y+j<h) tile[threadIdx.y+j][threadIdx.x]=src[size_t(y+j)*w+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(x<h&&y+j<w) dst[size_t(y+j)*h+x]=tile[threadIdx.x][threadIdx.y+j];
}
template<class T> __device__ void filterColumn(float *buf,int stride,int n,int x,Factors<T> c) {
    float first=buf[x],last=buf[size_t(n-1)*stride+x];
    float a=first*(c.B+c.b1+c.b2+c.b3);
    float b=buf[stride+x]*c.B+a*c.b1+first*(c.b2+c.b3);
    float v=buf[size_t(2)*stride+x]*c.B+b*c.b1+a*c.b2+first*c.b3;
    buf[x]=a;buf[stride+x]=b;buf[size_t(2)*stride+x]=v;
    for(int j=3;j<n;++j) {
        float next=buf[size_t(j)*stride+x]*c.B+v*c.b1+b*c.b2+a*c.b3;
        buf[size_t(j)*stride+x]=next;a=b;b=v;v=next;
    }
    float extra2=last+c.M[2][0]*(v-last)+c.M[2][1]*(b-last)+c.M[2][2]*(a-last);
    float extra1=last+c.M[1][0]*(v-last)+c.M[1][1]*(b-last)+c.M[1][2]*(a-last);
    v=last+c.M[0][0]*(v-last)+c.M[0][1]*(b-last)+c.M[0][2]*(a-last);
    b=c.B*b+c.b1*v+c.b2*extra1+c.b3*extra2;
    a=c.B*a+c.b1*b+c.b2*v+c.b3*extra1;
    buf[size_t(n-1)*stride+x]=v;buf[size_t(n-2)*stride+x]=b;buf[size_t(n-3)*stride+x]=a;
    float old=v;v=a;a=old;
    for(int j=n-4;j>=0;--j) {
        float next=buf[size_t(j)*stride+x]*c.B+v*c.b1+b*c.b2+a*c.b3;
        buf[size_t(j)*stride+x]=next;a=b;b=v;v=next;
    }
}
__global__ void filter(float *buf,int stride,int n,int vectorWidth,Coefficients c) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=stride) return;
    if(x<stride-stride%vectorWidth) filterColumn(buf,stride,n,x,c.f);
    else filterColumn(buf,stride,n,x,c.d);
}
inline void blur(const float *src,float *transposed,float *out,int w,int h,Coefficients c,cudaStream_t stream) {
    transpose<<<dim3((w+31)/32,(h+31)/32),dim3(32,8),0,stream>>>(src,transposed,w,h);
    filter<<<(h+127)/128,128,0,stream>>>(transposed,h,w,4,c);
    transpose<<<dim3((h+31)/32,(w+31)/32),dim3(32,8),0,stream>>>(transposed,out,h,w);
    filter<<<(w+127)/128,128,0,stream>>>(out,w,h,8,c);
}
} // namespace recursive
