// SPDX-License-Identifier: GPL-3.0-or-later
// Same coefficients as the ART CUDA experiment. Derived from gauss.cc.
// Original: Copyright (c) 2004-2010 Gabor Horvath.
#pragma once
#include <cmath>
namespace rtengine { namespace opencl {
struct Coefficients {
    float c00, c10, c11, c20, c21, c22, c30, c31, b0, b1;
};

Coefficients coefficients(double sigma, int radius)
{
    Coefficients c{};
    if (radius == 1) {
        const double inv = 1.0 / sigma;
        double c1 = std::exp(-0.5 * (inv * inv));
        double c2 = std::exp(-(inv * inv));
        const double sum = 1.0 + 4.0 * (c1 + c2);
        c.c00 = 1.0 / sum; c.c10 = c1 / sum; c.c11 = c2 / sum;
        double b1 = std::exp(-1.0 / (2.0 * sigma * sigma));
        const double bsum = 2.0 * b1 + 1.0;
        c.b0 = 1.0 / bsum; c.b1 = b1 / bsum;
        return c;
    }
    // Exactly the float accumulation and truncated support of compute5x5kernel
    // / compute7x7kernel in gauss.cc (including conversion of sigma to float).
    const float sf = sigma;
    const double temp = -2.f * (sf * sf);
    const double cutoff = 3.0 * (radius == 2 ? 0.84 : 1.15);
    float k[7][7] = {};
    float sum = 0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= cutoff * cutoff) {
                k[y + 3][x + 3] = std::exp((x * x + y * y) / temp);
                sum += k[y + 3][x + 3];
            }
        }
    }
    c.c00 = k[3][3] / sum; c.c10 = k[2][3] / sum;
    c.c11 = k[2][2] / sum; c.c20 = k[1][3] / sum;
    c.c21 = k[1][2] / sum; c.c22 = k[1][1] / sum;
    c.c30 = k[0][3] / sum; c.c31 = k[0][2] / sum;
    return c;
}

} }
