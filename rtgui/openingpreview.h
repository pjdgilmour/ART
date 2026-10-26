// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <gtkmm.h>
#include <future>
#include <memory>
#include <atomic>
class OpeningPreview final: public Gtk::Box {
public:
    OpeningPreview();
    ~OpeningPreview() override;
    void begin(const Glib::ustring &path);
    void finish(const Glib::ustring &path);
private:
    Gtk::Label caption_;
    Gtk::DrawingArea area_;
    Cairo::RefPtr<Cairo::ImageSurface> image_;
    std::future<Cairo::RefPtr<Cairo::ImageSurface>> pending_;
    std::shared_ptr<std::atomic<unsigned>> generation_;
    sigc::connection poll_;
    Glib::ustring path_;
    bool painted_=false;
    bool draw(const Cairo::RefPtr<Cairo::Context>&);
    bool poll();
};
