// SPDX-License-Identifier: GPL-3.0-or-later
#include "hardwaremonitor.h"
#include "multilangmgr.h"
#include "../rtengine/pipelinetime.h"
#include "../rtengine/threadpool.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#ifdef __linux__
#include <dlfcn.h>
#include <unistd.h>
#endif
namespace {
struct GpuProbe {
    int utilization=-1;
    unsigned long long used=0,total=0;
#ifdef __linux__
    // Stable NVML v1 ABI, loaded optionally; no CUDA/NVML build dependency.
    struct Memory {unsigned long long total,free,used;};
    struct Util {unsigned gpu,memory;};
    void *library=nullptr,*device=nullptr;
    int (*shutdown)()=nullptr;
    int (*getMemory)(void*,Memory*)=nullptr;
    int (*getUtil)(void*,Util*)=nullptr;
    bool initialized=false;
    GpuProbe() {
        rtengine::PipelineTimer timer("hardware_monitor_init");
        library=dlopen("libnvidia-ml.so.1",RTLD_LAZY|RTLD_LOCAL);
        if(!library) return;
        auto init=reinterpret_cast<int(*)()>(dlsym(library,"nvmlInit_v2"));
        auto handle=reinterpret_cast<int(*)(unsigned,void**)>(dlsym(library,"nvmlDeviceGetHandleByIndex_v2"));
        shutdown=reinterpret_cast<int(*)()>(dlsym(library,"nvmlShutdown"));
        getMemory=reinterpret_cast<int(*)(void*,Memory*)>(dlsym(library,"nvmlDeviceGetMemoryInfo"));
        getUtil=reinterpret_cast<int(*)(void*,Util*)>(dlsym(library,"nvmlDeviceGetUtilizationRates"));
        if(init && handle && shutdown && getMemory && getUtil && init()==0) {
            initialized=true;if(handle(0,&device)!=0) device=nullptr;
        }
    }
    ~GpuProbe() {if(initialized) shutdown();if(library) dlclose(library);}
    void sample() {
        utilization=-1;used=total=0;
        if(!device) return;
        Util u;Memory m;
        if(getUtil(device,&u)==0) utilization=u.gpu;
        if(getMemory(device,&m)==0) {used=m.used;total=m.total;}
    }
#else
    void sample() {}
#endif
};
}
HardwareMonitor::HardwareMonitor():Gtk::Box(Gtk::ORIENTATION_HORIZONTAL) {
    set_spacing(8);set_margin_start(8);set_margin_end(8);
    label_.set_xalign(.5f);label_.get_style_context()->add_class("dim-label");
    // Share the toolbar's free space without displacing its buttons.
    label_.set_ellipsize(Pango::ELLIPSIZE_END);
    label_.set_single_line_mode(true);
    pack_end(label_,true,true);
    set_tooltip_text(M("HARDWARE_MONITOR_TOOLTIP"));
    label_.set_text("CPU —   GPU —   RAM —   VRAM —");
}
void HardwareMonitor::on_map() {
    Gtk::Box::on_map();
    {std::lock_guard<std::mutex> lock(mutex_);mapped_=true;}
    if (!worker_.joinable()) {
        try {worker_=std::thread(&HardwareMonitor::sampleLoop,this);}
        catch(const std::system_error &) {return;}
    }
    wake_.notify_all();
    timer_=Glib::signal_timeout().connect(sigc::mem_fun(*this,&HardwareMonitor::refresh),1000);
}
void HardwareMonitor::on_unmap() {
    timer_.disconnect();
    {std::lock_guard<std::mutex> lock(mutex_);mapped_=false;}
    wake_.notify_all();
    Gtk::Box::on_unmap();
}
HardwareMonitor::~HardwareMonitor() {
    timer_.disconnect();
    {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}
    wake_.notify_all();if(worker_.joinable()) worker_.join();
}
bool HardwareMonitor::refresh() {
    rtengine::PipelineTimer timer("hardware_monitor_ui");
    std::string text;
    {std::lock_guard<std::mutex> lock(mutex_);text=text_;}
    const auto tasks=rtengine::ThreadPool::snapshot();
    const Glib::ustring status = text+"   "+M("HARDWARE_MONITOR_TASKS")+" "+std::to_string(tasks.active)+"/"+std::to_string(tasks.queued);
    if (label_.get_text() != status) {
        label_.set_text(status);
        label_.set_tooltip_text(status+"\n"+M("HARDWARE_MONITOR_TOOLTIP"));
    }
    return true;
}
void HardwareMonitor::sampleLoop() {
    GpuProbe gpu;
    unsigned long long previousTotal=0,previousIdle=0;
    for(;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!mapped_) previousTotal=previousIdle=0;
            wake_.wait(lock,[this]{return stop_ || mapped_;});
            if (stop_) break;
        }
        {
        rtengine::PipelineTimer timer("hardware_monitor_sample");
        double cpu=-1,ram=-1;
#ifdef __linux__
        std::ifstream stat("/proc/stat");std::string line;
        if(std::getline(stat,line)) {
            std::istringstream row(line);std::string name;row>>name;
            unsigned long long v[8]={},total=0;
            for(auto &x:v) {row>>x;total+=x;}
            auto idle=v[3]+v[4];
            if(previousTotal && total>previousTotal && idle>=previousIdle) cpu=100.*(1.-double(idle-previousIdle)/(total-previousTotal));
            previousTotal=total;previousIdle=idle;
        }
        std::ifstream mem("/proc/self/statm");unsigned long long pages=0,rss=0;
        if(mem>>pages>>rss) ram=double(rss)*sysconf(_SC_PAGESIZE)/(1024.*1024.*1024.);
#endif
        gpu.sample();
        std::ostringstream out;out<<std::fixed<<std::setprecision(0)<<"CPU ";
        if(cpu>=0) out<<cpu<<"%";else out<<"—";
        out<<"   GPU ";if(gpu.utilization>=0) out<<gpu.utilization<<"%";else out<<"—";
        out<<std::setprecision(2)<<"   RAM ";if(ram>=0) out<<ram<<" GiB";else out<<"—";
        out<<"   VRAM ";if(gpu.total) out<<double(gpu.used)/(1024.*1024.*1024.)<<" GiB";else out<<"—";
        {std::lock_guard<std::mutex> lock(mutex_);text_=out.str();}
        }
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait_for(lock,std::chrono::seconds(1),[this]{return stop_ || !mapped_;});
        if (stop_) break;
    }
}
