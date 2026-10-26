/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "improcfun.h"
#include "lanczos.h"
#include "pipelinetime.h"

#include "opthelper.h"
#include "rt_math.h"

//#define PROFILE

#ifdef PROFILE
#   include <iostream>
#endif


namespace rtengine {

void ImProcFunctions::Lanczos(Imagefloat *src, Imagefloat *dst, float scale)
{
    PipelineTimer timer("resize_lanczos", dst->getWidth(), dst->getHeight());
    auto mode = src->mode();
    src->setMode(Imagefloat::Mode::LAB, multiThread);
    dst->assignMode(Imagefloat::Mode::LAB);
    lanczosCpu(src->g.ptrs, src->r.ptrs, src->b.ptrs, src->getWidth(), src->getHeight(),
               dst->g.ptrs, dst->r.ptrs, dst->b.ptrs, dst->getWidth(), dst->getHeight(), scale);
    dst->setMode(mode, multiThread);
}


float ImProcFunctions::resizeScale(const ProcParams* params, int fw, int fh, int &imw, int &imh)
{
    imw = fw;
    imh = fh;

    if (!params || !params->resize.enabled) {
        return 1.0;
    }

    // get the resize parameters
    int refw, refh;
    double dScale;

    if (params->crop.enabled && params->resize.appliesTo == "Cropped area") {
        // the resize values applies to the crop dimensions
        refw = params->crop.w;
        refh = params->crop.h;
    } else {
        // the resize values applies to the image dimensions
        // if a crop exists, it will be resized to the calculated scale
        refw = fw;
        refh = fh;
    }

    int r_width = params->resize.get_width();
    int r_height = params->resize.get_height();

    switch (params->resize.dataspec) {
        case (1):
            // Width
            dScale = (double)r_width / (double)refw;
            break;

        case (2):
            // Height
            dScale = (double)r_height / (double)refh;
            break;

        case (3):

            // FitBox
            if ((double)refw / (double)refh > (double)r_width / (double)r_height) {
                dScale = (double)r_width / (double)refw;
            } else {
                dScale = (double)r_height / (double)refh;
            }
            dScale = (dScale > 1.0 && !params->resize.allowUpscaling) ? 1.0 : dScale;

            break;

        default:
            // Scale
            dScale = params->resize.scale;
            break;
    }

    if (fabs (dScale - 1.0) <= 1e-5) {
        return 1.0;
    }

    if (params->crop.enabled && params->resize.appliesTo == "Full image") {
        imw = params->crop.w;
        imh = params->crop.h;
    } else {
        imw = refw;
        imh = refh;
    }

    imw = (int) ( (double)imw * dScale + 0.5 );
    imh = (int) ( (double)imh * dScale + 0.5 );
    return (float)dScale;
}


void ImProcFunctions::resize(Imagefloat* src, Imagefloat* dst, float dScale)
{
#ifdef PROFILE
    time_t t1 = clock();
#endif

    Lanczos(src, dst, dScale);

#ifdef PROFILE
    time_t t2 = clock();
    std::cout << "Resize: "
              << (float) (t2 - t1) / CLOCKS_PER_SEC << std::endl;
#endif
}

} // namespace rtengine
