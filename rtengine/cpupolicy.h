// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vector>
#include <string>
namespace rtengine {
struct CpuTopology {
    std::vector<int> allowed, primary, foreground, background, ui;
    unsigned physical=1, logical=1, sockets=1, capacityClasses=0, cacheLine=0, numaNodes=0, largestCacheKiB=0, quotaThreads=0;
    bool known=false, sse2=false, sse42=false, avx=false, avx2=false, fma=false, avx512=false, sse41=false;
};
const CpuTopology &cpuTopology();
bool adaptiveScheduling();
int processingBudget();
int backgroundBudget();
void applyCpuPolicy(bool background=false, bool ui=false);
}
