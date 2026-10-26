// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdlib>
#include <cstring>
namespace rtengine {
// Experimental execution policies; keep the legacy path available for A/B tests.
inline bool performanceFlag(const char *name) {
    const char *value=std::getenv(name);
    return value && std::strcmp(value,"1")==0;
}
}
