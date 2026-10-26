// SPDX-License-Identifier: GPL-3.0-or-later
// Port of ART CPU/CUDA experiments. Original CPU: Copyright (c) 2004-2010 Gabor Horvath.
#pragma OPENCL FP_CONTRACT OFF
typedef struct {
    float c00, c10, c11, c20, c21, c22, c30, c31, b0, b1;
} Coefficients;
__kernel void initialize(__global float *lum, __global float *estimate, __global float *stopped, int n)
{
    int i = get_global_id(0);
    if (i < n) {
        lum[i] += 1000.f;
        estimate[i] = fmax(lum[i], 0.f);
        stopped[i] = NAN;
    }
}

inline float output(float original, float estimate, float blend, unsigned char impulse, float amount)
{
    if (isnan(estimate)) return original;
    const float b = impulse ? 0.f : blend * amount;
    return b * fmax(estimate, 0.f) + (1.f - b) * original;
}

inline float gaussian1(__global const float *s, int x, int y, int w, int h, Coefficients c)
{
    const int i = y * w + x;
    if (1 == 1) {
        if (y == 0 || y == h - 1) {
            if (x == 0 || x == w - 1) return s[i];
            return c.b1 * (s[i-1] + s[i+1]) + c.b0 * s[i];
        }
        if (x == 0 || x == w - 1) return c.b1 * (s[i-w] + s[i+w]) + c.b0 * s[i];
        return c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    if (1 == 2) {
        return c.c21 * (s[i-2*w-1] + s[i-2*w+1] + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
               c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
               c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    // Preserve the existing CPU's extra c21 factor on this one term. Changing
    // it would change saved edits; any correction belongs in a separate change.
    return c.c31 * (s[i-3*w-1] + s[i-3*w+1] + s[i-w-3] + s[i-w+3] + s[i+w-3] + s[i+w+3] + s[i+3*w-1] + s[i+3*w+1]) +
           c.c30 * (s[i-3*w] + s[i-3] + s[i+3] + s[i+3*w]) +
           c.c22 * (s[i-2*w-2] + s[i-2*w+2] + s[i+2*w-2] + s[i+2*w+2]) +
           c.c21 * (s[i-2*w-1] + s[i-2*w+1] * c.c21 + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
           c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
           c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
           c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
}

__kernel void divide1(__global const float *estimate, __global const float *lum, __global float *ratio, int w, int h, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    if (1 > 1 && (x < 1 || y < 1 || x >= w-1 || y >= h-1)) {
        ratio[i] = 1.f;
        return;
    }
    const float v = gaussian1(estimate, x, y, w, h, c);
    ratio[i] = 1 == 1 ? fmax(lum[i] / (v > 0.f ? v : 1.f), 0.f) : lum[i] / fmax(v, 0.00001f);
}

__kernel void multiply1(__global const float *ratio, __global const float *lum, __global const float *blend, __global const uchar *impulse,
                         __global float *estimate, __global float *stopped, int w, int h, float amount, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    float v = estimate[i];
    if (1 == 1 || (x >= 1 && y >= 1 && x < w-1 && y < h-1)) {
        v *= gaussian1(ratio, x, y, w, h, c);
        estimate[i] = v;
    }
    if (isnan(stopped[i]) && fabs(v - lum[i]) > lum[i] * 0.2f) {
        stopped[i] = output(lum[i], v, blend[i], impulse[i], amount);
    }
}

inline float gaussian2(__global const float *s, int x, int y, int w, int h, Coefficients c)
{
    const int i = y * w + x;
    if (2 == 1) {
        if (y == 0 || y == h - 1) {
            if (x == 0 || x == w - 1) return s[i];
            return c.b1 * (s[i-1] + s[i+1]) + c.b0 * s[i];
        }
        if (x == 0 || x == w - 1) return c.b1 * (s[i-w] + s[i+w]) + c.b0 * s[i];
        return c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    if (2 == 2) {
        return c.c21 * (s[i-2*w-1] + s[i-2*w+1] + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
               c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
               c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    // Preserve the existing CPU's extra c21 factor on this one term. Changing
    // it would change saved edits; any correction belongs in a separate change.
    return c.c31 * (s[i-3*w-1] + s[i-3*w+1] + s[i-w-3] + s[i-w+3] + s[i+w-3] + s[i+w+3] + s[i+3*w-1] + s[i+3*w+1]) +
           c.c30 * (s[i-3*w] + s[i-3] + s[i+3] + s[i+3*w]) +
           c.c22 * (s[i-2*w-2] + s[i-2*w+2] + s[i+2*w-2] + s[i+2*w+2]) +
           c.c21 * (s[i-2*w-1] + s[i-2*w+1] * c.c21 + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
           c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
           c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
           c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
}

__kernel void divide2(__global const float *estimate, __global const float *lum, __global float *ratio, int w, int h, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    if (2 > 1 && (x < 2 || y < 2 || x >= w-2 || y >= h-2)) {
        ratio[i] = 1.f;
        return;
    }
    const float v = gaussian2(estimate, x, y, w, h, c);
    ratio[i] = 2 == 1 ? fmax(lum[i] / (v > 0.f ? v : 1.f), 0.f) : lum[i] / fmax(v, 0.00001f);
}

__kernel void multiply2(__global const float *ratio, __global const float *lum, __global const float *blend, __global const uchar *impulse,
                         __global float *estimate, __global float *stopped, int w, int h, float amount, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    float v = estimate[i];
    if (2 == 1 || (x >= 2 && y >= 2 && x < w-2 && y < h-2)) {
        v *= gaussian2(ratio, x, y, w, h, c);
        estimate[i] = v;
    }
    if (isnan(stopped[i]) && fabs(v - lum[i]) > lum[i] * 0.2f) {
        stopped[i] = output(lum[i], v, blend[i], impulse[i], amount);
    }
}

inline float gaussian3(__global const float *s, int x, int y, int w, int h, Coefficients c)
{
    const int i = y * w + x;
    if (3 == 1) {
        if (y == 0 || y == h - 1) {
            if (x == 0 || x == w - 1) return s[i];
            return c.b1 * (s[i-1] + s[i+1]) + c.b0 * s[i];
        }
        if (x == 0 || x == w - 1) return c.b1 * (s[i-w] + s[i+w]) + c.b0 * s[i];
        return c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    if (3 == 2) {
        return c.c21 * (s[i-2*w-1] + s[i-2*w+1] + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
               c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
               c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
               c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
    }
    // Preserve the existing CPU's extra c21 factor on this one term. Changing
    // it would change saved edits; any correction belongs in a separate change.
    return c.c31 * (s[i-3*w-1] + s[i-3*w+1] + s[i-w-3] + s[i-w+3] + s[i+w-3] + s[i+w+3] + s[i+3*w-1] + s[i+3*w+1]) +
           c.c30 * (s[i-3*w] + s[i-3] + s[i+3] + s[i+3*w]) +
           c.c22 * (s[i-2*w-2] + s[i-2*w+2] + s[i+2*w-2] + s[i+2*w+2]) +
           c.c21 * (s[i-2*w-1] + s[i-2*w+1] * c.c21 + s[i-w-2] + s[i-w+2] + s[i+w-2] + s[i+w+2] + s[i+2*w-1] + s[i+2*w+1]) +
           c.c20 * (s[i-2*w] + s[i-2] + s[i+2] + s[i+2*w]) +
           c.c11 * (s[i-w-1] + s[i-w+1] + s[i+w-1] + s[i+w+1]) +
           c.c10 * (s[i-w] + s[i-1] + s[i+1] + s[i+w]) + c.c00 * s[i];
}

__kernel void divide3(__global const float *estimate, __global const float *lum, __global float *ratio, int w, int h, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    if (3 > 1 && (x < 3 || y < 3 || x >= w-3 || y >= h-3)) {
        ratio[i] = 1.f;
        return;
    }
    const float v = gaussian3(estimate, x, y, w, h, c);
    ratio[i] = 3 == 1 ? fmax(lum[i] / (v > 0.f ? v : 1.f), 0.f) : lum[i] / fmax(v, 0.00001f);
}

__kernel void multiply3(__global const float *ratio, __global const float *lum, __global const float *blend, __global const uchar *impulse,
                         __global float *estimate, __global float *stopped, int w, int h, float amount, Coefficients c)
{
    const int i = get_global_id(0);
    if (i >= w * h) return;
    const int x = i % w, y = i / w;
    float v = estimate[i];
    if (3 == 1 || (x >= 3 && y >= 3 && x < w-3 && y < h-3)) {
        v *= gaussian3(ratio, x, y, w, h, c);
        estimate[i] = v;
    }
    if (isnan(stopped[i]) && fabs(v - lum[i]) > lum[i] * 0.2f) {
        stopped[i] = output(lum[i], v, blend[i], impulse[i], amount);
    }
}

__kernel void finish(__global const float *lum, __global const float *blend, __global const uchar *impulse,
                       __global const float *estimate, __global const float *stopped, __global float *result, int n, float amount)
{
    const int i = get_global_id(0);
    if (i < n) {
        float v = stopped[i];
        if (isnan(v)) v = output(lum[i], estimate[i], blend[i], impulse[i], amount);
        result[i] = fmax(v - 1000.f, 0.f);
    }
}

__kernel void vertical(__global const float *src, __global float *tmp, int sw, int sh, int dh,
                         __global const float *weights, __global const int *first, __global const int *last, int support)
{
    size_t i = get_global_id(0);
    if (i >= ((size_t)(sw))*dh*3) return;
    int x = i % sw, y = (i/sw) % dh, c = i/(((size_t)(sw))*dh);
    float v = 0;
    for (int k = first[y]; k < last[y]; ++k)
        v += weights[((size_t)(y))*support+k-first[y]] * src[(((size_t)(c))*sh+k)*sw+x];
    tmp[i] = v;
}
__kernel void horizontal(__global const float *tmp, __global float *out, int sw, int dw, int dh,
                           __global const float *weights, __global const int *first, __global const int *last, int support)
{
    size_t i = get_global_id(0);
    if (i >= ((size_t)(dw))*dh*3) return;
    int x = i % dw, y = (i/dw) % dh, c = i/(((size_t)(dw))*dh);
    float v = 0;
    for (int k = first[x]; k < last[x]; ++k)
        v += weights[((size_t)(x))*support+k-first[x]] * tmp[(((size_t)(c))*dh+y)*sw+k];
    out[i] = v;
}