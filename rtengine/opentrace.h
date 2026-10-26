// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace rtengine {
inline void openTrace(const char *event) {
    static const bool enabled = [] {const char *p=std::getenv("ART_OPEN_TRACE");return p && !std::strcmp(p,"1");}();
    if(enabled) {
        const auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr,"ART_OPEN_TRACE {\"event\":\"%s\",\"us\":%lld}\n",event,static_cast<long long>(us));
    }
}
}
