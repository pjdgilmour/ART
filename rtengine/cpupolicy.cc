// SPDX-License-Identifier: GPL-3.0-or-later
#include "cpupolicy.h"
#include <thread>
#include <fstream>
#include <set>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#ifdef __linux__
#include <sched.h>
#include <dirent.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
namespace rtengine {
namespace {
int readInt(const std::string &path,int fallback) {std::ifstream f(path);int v;return f>>v?v:fallback;}
CpuTopology detect() {
    CpuTopology t;
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if(sched_getaffinity(0,sizeof(mask),&mask)==0) for(int i=0;i<CPU_SETSIZE;++i) if(CPU_ISSET(i,&mask)) t.allowed.push_back(i);
#endif
    if(t.allowed.empty()) for(unsigned i=0;i<std::max(1u,std::thread::hardware_concurrency());++i) t.allowed.push_back(i);
    t.logical=t.allowed.size();
    std::map<std::pair<int,int>,std::vector<int>> cores;
    std::set<int> sockets,capacities,nodes;
#ifdef __linux__
    t.known=true;
    for(int cpu:t.allowed) {
        const std::string p="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/";
        int socket=readInt(p+"topology/physical_package_id",-1),core=readInt(p+"topology/core_id",-1);
        if(core<0||socket<0) {t.known=false;core=cpu;socket=0;}
        cores[{socket,core}].push_back(cpu);sockets.insert(socket);
        int capacity=readInt(p+"cpu_capacity",0);if(capacity) capacities.insert(capacity);
        if(!t.cacheLine) t.cacheLine=readInt(p+"cache/index0/coherency_line_size",0);
        if (DIR *dir=opendir(p.c_str())) {
            while (auto entry=readdir(dir)) {
                if (!std::strncmp(entry->d_name,"node",4) && entry->d_name[4]>='0' && entry->d_name[4]<='9')
                    nodes.insert(std::atoi(entry->d_name+4));
            }
            closedir(dir);
        }
        for (int index=0;index<16;++index) {
            std::ifstream size(p+"cache/index"+std::to_string(index)+"/size");
            unsigned n=0;char unit='K';
            if (size>>n>>unit) t.largestCacheKiB=std::max(t.largestCacheKiB,n*(unit=='M'?1024u:1u));
        }
    }
#else
    for(int cpu:t.allowed) cores[{0,cpu}].push_back(cpu);
#endif
    t.numaNodes=nodes.size();
#ifdef __linux__
    // Respect the tightest visible cgroup-v2 CPU quota, including ancestors.
    // Affinity/cpuset constraints have already been applied above.
    std::string group;
    std::ifstream membership("/proc/self/cgroup");
    while (std::getline(membership,group)) if (group.compare(0,3,"0::")==0) {
        group=group.substr(3);
        if (group.empty() || group[0]!='/' || group.find("..")!=std::string::npos) break;
        for (;;) {
            std::ifstream quota("/sys/fs/cgroup"+group+"/cpu.max");
            std::string amount;unsigned long long period=0;
            if (quota>>amount>>period && amount!="max" && period) {
                const auto value=std::strtoull(amount.c_str(),nullptr,10);
                if (value) {
                    unsigned threads=std::min<unsigned long long>(t.logical,1+(value-1)/period);
                    t.quotaThreads=t.quotaThreads?std::min(t.quotaThreads,threads):threads;
                }
            }
            if (group.empty()) break;
            group=group.substr(0,group.find_last_of('/'));
        }
        break;
    }
#endif
    t.physical=cores.size();t.sockets=std::max(size_t(1),sockets.size());t.capacityClasses=capacities.size();
    for(const auto &c:cores) t.primary.push_back(c.second.front());
    // Groups are an explicit experiment. Reserve complete physical cores,
    // including their SMT siblings; never assume CPU 0 is in the allowed mask.
    const unsigned bg=t.physical>=8?std::min(8u,std::max(2u,t.physical/4)):0;
    unsigned i=0;
    for(const auto &c:cores) {
        auto &group=bg && i==0?t.ui:(bg && i<=bg?t.background:t.foreground);
        group.insert(group.end(),c.second.begin(),c.second.end());++i;
    }
    if(t.foreground.empty()) t.foreground=t.allowed;
    if(t.background.empty()) t.background=t.allowed;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();t.sse2=__builtin_cpu_supports("sse2");t.sse42=__builtin_cpu_supports("sse4.2");t.avx=__builtin_cpu_supports("avx");t.avx2=__builtin_cpu_supports("avx2");t.fma=__builtin_cpu_supports("fma");
    t.avx512=__builtin_cpu_supports("avx512f");t.sse41=__builtin_cpu_supports("sse4.1");
#endif
    return t;
}
}
const CpuTopology &cpuTopology() {static const CpuTopology t=detect();return t;}
bool adaptiveScheduling() {const char *p=std::getenv("ART_SCHEDULER");return p && (!std::strcmp(p,"dynamic")||!std::strcmp(p,"groups"));}
int processingBudget() {
    const auto &t=cpuTopology();
    int n=t.known?t.physical:t.logical;
    const char *p=std::getenv("OMP_NUM_THREADS");
    if(p && std::atoi(p)>0) n=std::min(int(t.logical),std::atoi(p));
    else if(adaptiveScheduling() && n>2) --n;
    const char *s=std::getenv("ART_SCHEDULER");
    if(s && !std::strcmp(s,"groups") && !t.ui.empty()) n=std::min(n,int(t.physical)-1-backgroundBudget());
    if(t.quotaThreads) n=std::min(n,int(t.quotaThreads));
    return std::max(1,n);
}
int backgroundBudget() {
    const auto &t=cpuTopology();
    const unsigned available=t.quotaThreads?std::min(t.physical,t.quotaThreads):t.physical;
    return std::max(1u,std::min(8u,available/4));
}
void applyCpuPolicy(bool background,bool ui) {
    if(!adaptiveScheduling()) return;
#ifdef _OPENMP
    omp_set_num_threads(background?1:processingBudget());
#endif
#ifdef __linux__
    const char *s=std::getenv("ART_SCHEDULER");
    if(s && !std::strcmp(s,"groups")) {
        const auto &t=cpuTopology();
        const auto &cpus=ui?t.ui:(background?t.background:t.foreground);
        if(!cpus.empty()) {
            cpu_set_t mask;CPU_ZERO(&mask);
            for(int c:cpus) if(c<CPU_SETSIZE) CPU_SET(c,&mask);
            sched_setaffinity(0,sizeof(mask),&mask);
        }
    }
#endif
}
}
