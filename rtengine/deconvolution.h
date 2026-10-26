// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

namespace rtengine {
void deconvsharpening(float **luminance, float **blend, char **impulse,
                     int width, int height, double sigma, float amount, bool multiThread,
                     bool forceCPU = false);
// Independent center/corner estimates; experimental CPU/GPU overlap is opt-in.
void deconvsharpeningPair(float **center, float **corner, float **blend, char **impulse,
                         int width, int height, double sigma, double cornerSigma,
                         float amount, bool multiThread);
}
