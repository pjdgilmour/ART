// SPDX-License-Identifier: GPL-3.0-or-later
#include "openingpreview.h"
#include "../rtengine/previewimage.h"
#include "../rtengine/threadpool.h"
#include "../rtengine/performance.h"
#include "../rtengine/opentrace.h"
#include "multilangmgr.h"
#include <glib/gstdio.h>
#include <list>
#include <mutex>
#include <algorithm>
namespace {
using Surface=Cairo::RefPtr<Cairo::ImageSurface>;
struct Entry { std::string path; gint64 size,mtime; Surface image; size_t bytes; };
std::mutex cacheMutex;
std::list<Entry> cache;
Surface loadPreview(const Glib::ustring &path) {
    GStatBuf st;
    if(g_stat(path.c_str(),&st)!=0) return Surface();
    gint64 stamp=st.st_mtime;
#ifdef __linux__
    stamp=stamp*1000000000LL+st.st_mtim.tv_nsec;
#endif
    if(rtengine::performanceFlag("ART_PREVIEW_CACHE")) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for(auto i=cache.begin();i!=cache.end();++i) if(i->path==path.raw() && i->size==st.st_size && i->mtime==stamp) {
            Surface result=i->image;cache.splice(cache.begin(),cache,i);rtengine::openTrace("camera_cache_hit");return result;
        }
    }
    const auto dot=path.raw().find_last_of('.');
    if(dot==std::string::npos) return Surface();
    rtengine::PreviewImage preview(path,path.substr(dot+1),1280,854,false,false,true);
    Surface image=preview.getImage();
    if(image && rtengine::performanceFlag("ART_PREVIEW_CACHE")) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cache.remove_if([&](const Entry &e){return e.path==path.raw();});
        cache.push_front({path.raw(),st.st_size,stamp,image,size_t(image->get_stride())*image->get_height()});
        size_t bytes=0;for(const auto &e:cache) bytes+=e.bytes;
        while(cache.size()>4 || bytes>24*1024*1024) {bytes-=cache.back().bytes;cache.pop_back();}
    }
    return image;
}
}
OpeningPreview::OpeningPreview():Gtk::Box(Gtk::ORIENTATION_VERTICAL),generation_(new std::atomic<unsigned>(0)) {
    caption_.set_text(M("OPENING_CAMERA_PREVIEW"));
    pack_start(caption_,false,false);pack_start(area_,true,true);
    area_.set_size_request(640,426);
    area_.signal_draw().connect(sigc::mem_fun(*this,&OpeningPreview::draw));
    get_style_context()->add_class("app-notification");
    set_halign(Gtk::ALIGN_CENTER);set_valign(Gtk::ALIGN_CENTER);
    set_no_show_all(true);
}
OpeningPreview::~OpeningPreview() {++*generation_;poll_.disconnect();}
void OpeningPreview::begin(const Glib::ustring &path) {
    path_=path;image_.clear();painted_=false;hide();poll_.disconnect();
    const unsigned version=++*generation_;auto generation=generation_;
    pending_=rtengine::ThreadPool::add_task(rtengine::ThreadPool::Priority::HIGH,[path,generation,version]() {
        if(generation->load()!=version) return Surface();
        rtengine::applyCpuPolicy(true);
        return loadPreview(path);
    });
    poll_=Glib::signal_timeout().connect(sigc::mem_fun(*this,&OpeningPreview::poll),25);
}
bool OpeningPreview::poll() {
    if(!pending_.valid() || pending_.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return true;
    try {image_=pending_.get();} catch(...) {image_.clear();}
    if(image_) {show();caption_.show();area_.show();queue_draw();rtengine::openTrace("camera_preview_ready");}
    return false;
}
bool OpeningPreview::draw(const Cairo::RefPtr<Cairo::Context> &cr) {
    if(!image_) return true;
    double w=area_.get_allocated_width(),h=area_.get_allocated_height();
    cr->set_source_rgb(.08,.08,.08);cr->paint();
    double scale=std::min(w/image_->get_width(),h/image_->get_height());
    cr->save();cr->translate((w-image_->get_width()*scale)/2,(h-image_->get_height()*scale)/2);cr->scale(scale,scale);
    cr->set_source(image_,0,0);cr->paint();cr->restore();
    if(!painted_) {painted_=true;rtengine::openTrace("camera_preview_painted");}
    return true;
}
void OpeningPreview::finish(const Glib::ustring &path) {
    if(path!=path_) return;
    path_.clear();
    ++*generation_;poll_.disconnect();hide();image_.clear();
    if(rtengine::performanceFlag("ART_PREFETCH_PREVIEW")) {
        auto generation=generation_;const unsigned version=generation->load();
        rtengine::ThreadPool::add_task(rtengine::ThreadPool::Priority::LOWEST,[path,generation,version]() {
            if(generation->load()!=version) return;
            try {
                const auto folder=Glib::path_get_dirname(path);const Glib::ustring file=Glib::path_get_basename(path);
                const auto dot=file.raw().find_last_of('.');if(dot==std::string::npos) return;
                const auto extension=file.substr(dot).lowercase();std::vector<Glib::ustring> names;
                Glib::Dir dir(folder);
                for(const auto &name:dir) {const auto p=name.find_last_of('.');if(p!=std::string::npos && Glib::ustring(name.substr(p)).lowercase()==extension) names.emplace_back(name);}
                std::sort(names.begin(),names.end());auto next=std::upper_bound(names.begin(),names.end(),file);
                if(next!=names.end() && generation->load()==version) {loadPreview(Glib::build_filename(folder,*next));rtengine::openTrace("adjacent_preview_cached");}
            } catch(...) {}
        });
    }
}
