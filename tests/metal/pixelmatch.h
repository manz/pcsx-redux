/*
 * C++ port of https://github.com/mapbox/pixelmatch
 * Ported from the pure Python version in maparazzo/tests/pixelmatch.py
 *
 * Operates on flat RGBA8 buffers (4 bytes per pixel).
 * Returns mismatch count and optionally fills a diff image.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pixelmatch {

struct Options {
    float threshold = 0.1f;
    bool includeAA = false;
    float alpha = 0.1f;
    uint8_t aaColor[3] = {255, 255, 0};
    uint8_t diffColor[3] = {255, 0, 0};
    bool diffMask = false;
};

namespace detail {

inline void drawPixel(uint8_t *out, int pos, uint8_t r, uint8_t g, uint8_t b) {
    out[pos + 0] = r;
    out[pos + 1] = g;
    out[pos + 2] = b;
    out[pos + 3] = 255;
}

inline void drawGrayPixel(const uint8_t *img, int i, float alpha, uint8_t *out) {
    float r = img[i], g = img[i + 1], b = img[i + 2], a = img[i + 3];
    float gray = r * 0.29889531f + g * 0.58662247f + b * 0.11448223f;
    float ea = alpha * (a / 255.0f);
    int val = std::clamp(static_cast<int>(255.0f + (gray - 255.0f) * ea), 0, 255);
    drawPixel(out, i, val, val, val);
}

inline float colorDelta(const uint8_t *img1, const uint8_t *img2, int k, int m, bool yOnly) {
    float r1 = img1[k], g1 = img1[k + 1], b1 = img1[k + 2], a1 = img1[k + 3];
    float r2 = img2[m], g2 = img2[m + 1], b2 = img2[m + 2], a2 = img2[m + 3];

    float dr = r1 - r2, dg = g1 - g2, db = b1 - b2, da = a1 - a2;
    if (dr == 0 && dg == 0 && db == 0 && da == 0) return 0.0f;

    if (a1 < 255.0f || a2 < 255.0f) {
        constexpr float phi = 1.618033988749895f;
        constexpr float phiSq = 2.618033988749895f;
        int rbf = k % 2;
        int gbf = static_cast<int>(std::trunc(k / phi)) % 2;
        int bbf = static_cast<int>(std::trunc(k / phiSq)) % 2;
        float bgR = 48.0f + 159.0f * rbf;
        float bgG = 48.0f + 159.0f * gbf;
        float bgB = 48.0f + 159.0f * bbf;
        dr = (r1 * a1 - r2 * a2 - bgR * da) / 255.0f;
        dg = (g1 * a1 - g2 * a2 - bgG * da) / 255.0f;
        db = (b1 * a1 - b2 * a2 - bgB * da) / 255.0f;
    }

    float y = dr * 0.29889531f + dg * 0.58662247f + db * 0.11448223f;
    if (yOnly) return y;

    float i = dr * 0.59597799f - dg * 0.27417610f - db * 0.32180189f;
    float q = dr * 0.21147017f - dg * 0.52261711f + db * 0.31114694f;
    float deltaSq = 0.5053f * y * y + 0.299f * i * i + 0.1957f * q * q;
    return y > 0 ? -deltaSq : deltaSq;
}

inline bool hasManySiblings(const uint32_t *img32, int x1, int y1, int w, int h) {
    int x0 = std::max(x1 - 1, 0), y0 = std::max(y1 - 1, 0);
    int x2 = std::min(x1 + 1, w - 1), y2 = std::min(y1 + 1, h - 1);
    uint32_t center = img32[y1 * w + x1];
    int zeroes = (x1 == x0 || x1 == x2 || y1 == y0 || y1 == y2) ? 1 : 0;
    for (int yy = y0; yy <= y2; yy++)
        for (int xx = x0; xx <= x2; xx++) {
            if (xx == x1 && yy == y1) continue;
            if (center == img32[yy * w + xx]) zeroes++;
            if (zeroes > 2) return true;
        }
    return false;
}

inline bool antialiased(const uint8_t *img, int x1, int y1, int w, int h,
                        const uint32_t *a32, const uint32_t *b32) {
    int x0 = std::max(x1 - 1, 0), y0 = std::max(y1 - 1, 0);
    int x2 = std::min(x1 + 1, w - 1), y2 = std::min(y1 + 1, h - 1);
    int cbi = (y1 * w + x1) * 4;
    int zeroes = (x1 == x0 || x1 == x2 || y1 == y0 || y1 == y2) ? 1 : 0;
    float minDelta = 0, maxDelta = 0;
    int minX = 0, minY = 0, maxX = 0, maxY = 0;

    for (int yy = y0; yy <= y2; yy++)
        for (int xx = x0; xx <= x2; xx++) {
            if (xx == x1 && yy == y1) continue;
            int nbi = (yy * w + xx) * 4;
            float d = colorDelta(img, img, cbi, nbi, true);
            if (d == 0) {
                zeroes++;
                if (zeroes > 2) return false;
            } else if (d < minDelta) {
                minDelta = d;
                minX = xx;
                minY = yy;
            } else if (d > maxDelta) {
                maxDelta = d;
                maxX = xx;
                maxY = yy;
            }
        }
    if (minDelta == 0 || maxDelta == 0) return false;
    return (hasManySiblings(a32, minX, minY, w, h) && hasManySiblings(b32, minX, minY, w, h)) ||
           (hasManySiblings(a32, maxX, maxY, w, h) && hasManySiblings(b32, maxX, maxY, w, h));
}

}  // namespace detail

// Compare two RGBA8 images. Returns number of mismatched pixels.
// If diffOut is non-null, it must point to w*h*4 bytes and will be filled with the diff image.
inline int compare(const uint8_t *img1, const uint8_t *img2, int w, int h,
                   uint8_t *diffOut = nullptr, const Options &opts = {}) {
    float maxSqDelta = 35215.0f * opts.threshold * opts.threshold;
    int mismatch = 0;

    const auto *a32 = reinterpret_cast<const uint32_t *>(img1);
    const auto *b32 = reinterpret_cast<const uint32_t *>(img2);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int pos = idx * 4;

            float delta = 0;
            if (a32[idx] != b32[idx]) {
                delta = detail::colorDelta(img1, img2, pos, pos, false);
            }

            if (std::fabs(delta) > maxSqDelta) {
                bool isAA = detail::antialiased(img1, x, y, w, h, a32, b32) ||
                            detail::antialiased(img2, x, y, w, h, b32, a32);

                if (!opts.includeAA && isAA) {
                    if (diffOut && !opts.diffMask)
                        detail::drawPixel(diffOut, pos, opts.aaColor[0], opts.aaColor[1], opts.aaColor[2]);
                } else {
                    if (diffOut)
                        detail::drawPixel(diffOut, pos, opts.diffColor[0], opts.diffColor[1], opts.diffColor[2]);
                    mismatch++;
                }
            } else if (diffOut && !opts.diffMask) {
                detail::drawGrayPixel(img1, pos, opts.alpha, diffOut);
            }
        }
    }
    return mismatch;
}

// Convenience: compute mismatch percentage
inline float comparePercent(const uint8_t *img1, const uint8_t *img2, int w, int h,
                            uint8_t *diffOut = nullptr, const Options &opts = {}) {
    int mismatches = compare(img1, img2, w, h, diffOut, opts);
    return (w * h > 0) ? (static_cast<float>(mismatches) / (w * h)) * 100.0f : 0.0f;
}

}  // namespace pixelmatch
