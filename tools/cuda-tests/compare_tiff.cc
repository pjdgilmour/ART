// SPDX-License-Identifier: GPL-3.0-or-later
// Compare decoded uint16 samples, ignoring metadata and TIFF container layout.
#include <tiffio.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 3) { std::fprintf(stderr, "Usage: compare_tiff reference.tif candidate.tif\n"); return 2; }
    try {
        using Image = std::unique_ptr<TIFF, decltype(&TIFFClose)>;
        Image a(TIFFOpen(argv[1], "r"), TIFFClose), b(TIFFOpen(argv[2], "r"), TIFFClose);
        if (!a || !b) throw std::runtime_error("cannot open TIFF");
        uint32_t w[2], h[2];
        uint16_t samples[2];
        TIFF *images[] = {a.get(), b.get()};
        for (int i = 0; i < 2; ++i) {
            uint16_t bits, planar, format;
            if (!TIFFGetField(images[i], TIFFTAG_IMAGEWIDTH, &w[i]) ||
                !TIFFGetField(images[i], TIFFTAG_IMAGELENGTH, &h[i])) throw std::runtime_error("missing dimensions");
            TIFFGetFieldDefaulted(images[i], TIFFTAG_SAMPLESPERPIXEL, &samples[i]);
            TIFFGetFieldDefaulted(images[i], TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(images[i], TIFFTAG_PLANARCONFIG, &planar);
            TIFFGetFieldDefaulted(images[i], TIFFTAG_SAMPLEFORMAT, &format);
            if (TIFFIsTiled(images[i]) || bits != 16 || planar != PLANARCONFIG_CONTIG || format != SAMPLEFORMAT_UINT)
                throw std::runtime_error("requires stripped, contiguous, unsigned 16-bit TIFF");
        }
        if (w[0] != w[1] || h[0] != h[1] || samples[0] != samples[1]) throw std::runtime_error("shape mismatch");
        const size_t row = size_t(w[0]) * samples[0];
        if (TIFFScanlineSize(a.get()) != tmsize_t(row * 2) || TIFFScanlineSize(b.get()) != tmsize_t(row * 2))
            throw std::runtime_error("unexpected scanline size");
        std::vector<uint16_t> ar(row), br(row);
        uint64_t different = 0, sum = 0;
        int maximum = 0;
        for (uint32_t y = 0; y < h[0]; ++y) {
            if (TIFFReadScanline(a.get(), ar.data(), y) < 0 || TIFFReadScanline(b.get(), br.data(), y) < 0)
                throw std::runtime_error("cannot decode scanline");
            for (size_t x = 0; x < row; ++x) {
                int d = std::abs(int(ar[x]) - int(br[x]));
                maximum = std::max(maximum, d); sum += d; different += d != 0;
            }
        }
        std::printf("%ux%u channels=%u different=%llu max_lsb=%d mean_lsb=%.12g\n", w[0], h[0], samples[0],
                    static_cast<unsigned long long>(different), maximum, double(sum) / (double(row) * h[0]));
        return different ? 1 : 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what()); return 2;
    }
}
