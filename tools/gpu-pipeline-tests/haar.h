// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
namespace experiment {
// Standalone resident-pyramid experiment. Not dispatched by the ART engine.
class Haar {
public:
    Haar(int width, int height, int levels);
    ~Haar();
    Haar(const Haar&)=delete;
    Haar& operator=(const Haar&)=delete;
    void upload(const float *input);
    void run();
    void download(float *output);
    size_t bytes() const;
private:
    int w_, h_, levels_;
    float *device_=nullptr;
};
}
