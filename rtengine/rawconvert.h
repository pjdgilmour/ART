// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
namespace rtengine {
// Extract alternating Bayer channels from interleaved RGGB-sized pixel records.
// Exact uint16 -> FP32 conversion; AVX2 is selected only after runtime detection.
void unpackBayerRow(const uint16_t *src,float *dst,int width,int even,int odd,bool simd);
}
