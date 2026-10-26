// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <gtkmm.h>
#include <thread>
#include <mutex>
#include <condition_variable>
class HardwareMonitor final: public Gtk::Box {
public:
    HardwareMonitor();
    ~HardwareMonitor() override;
private:
    Gtk::Label label_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_=false;
    bool mapped_=false;
    std::string text_;
    sigc::connection timer_;
    void on_map() override;
    void on_unmap() override;
    bool refresh();
    void sampleLoop();
};
