// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
namespace rtengine {
// Planar FP32, arbitrary row stride. Same vertical-then-horizontal order as ART.
void lanczosCpu(float *const *sL, float *const *sa, float *const *sb,
                int sW, int sH, float *const *dL, float *const *da, float *const *db,
                int dW, int dH, float scale);
// Coefficients shared with the CUDA experiment (no GPU sin approximation).
void lanczosWeights(int source, int dest, float scale, int support,
                    float *weights, int *first, int *last);
}
