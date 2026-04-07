/*
 * Unit tests for the pixelmatch algorithm (C++ port).
 * Validates color delta, anti-aliasing detection, and overall comparison.
 *
 * Build:
 *   c++ -std=c++2b -Itests/metal \
 *       -Ithird_party/googletest/googletest/include -Ithird_party/googletest/googletest \
 *       third_party/googletest/googletest/src/gtest-all.cc \
 *       tests/metal/pixelmatch_test.cc -o bins/pixelmatch_test
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "pixelmatch.h"

// Helper: fill an RGBA buffer with a solid color
static void fillRGBA(uint8_t *buf, int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    for (int i = 0; i < w * h; i++) {
        buf[i * 4 + 0] = r;
        buf[i * 4 + 1] = g;
        buf[i * 4 + 2] = b;
        buf[i * 4 + 3] = a;
    }
}

// Helper: set a single pixel
static void setPixel(uint8_t *buf, int w, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    int i = (y * w + x) * 4;
    buf[i] = r; buf[i + 1] = g; buf[i + 2] = b; buf[i + 3] = a;
}

TEST(PixelmatchTest, IdenticalImages_ZeroMismatches) {
    const int w = 16, h = 16;
    std::vector<uint8_t> img(w * h * 4);
    fillRGBA(img.data(), w, h, 128, 64, 200);

    int mismatches = pixelmatch::compare(img.data(), img.data(), w, h);
    EXPECT_EQ(mismatches, 0);
}

TEST(PixelmatchTest, AllBlack_ZeroMismatches) {
    const int w = 8, h = 8;
    std::vector<uint8_t> a(w * h * 4, 0), b(w * h * 4, 0);
    // Set alpha to 255 for both
    for (int i = 0; i < w * h; i++) { a[i * 4 + 3] = 255; b[i * 4 + 3] = 255; }

    EXPECT_EQ(pixelmatch::compare(a.data(), b.data(), w, h), 0);
}

TEST(PixelmatchTest, SinglePixelDiff_DetectedAsMismatch) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4);
    fillRGBA(a.data(), w, h, 100, 100, 100);
    fillRGBA(b.data(), w, h, 100, 100, 100);

    // Make one pixel vastly different
    setPixel(b.data(), w, 2, 2, 255, 0, 0);

    int mismatches = pixelmatch::compare(a.data(), b.data(), w, h);
    EXPECT_GE(mismatches, 1) << "Should detect at least the changed pixel";
}

TEST(PixelmatchTest, SmallDifference_BelowThreshold) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4);
    fillRGBA(a.data(), w, h, 100, 100, 100);
    fillRGBA(b.data(), w, h, 100, 100, 100);

    // 1-unit difference should be below default threshold
    setPixel(b.data(), w, 1, 1, 101, 100, 100);

    EXPECT_EQ(pixelmatch::compare(a.data(), b.data(), w, h), 0);
}

TEST(PixelmatchTest, SmallDifference_DetectedWithStrictThreshold) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4);
    fillRGBA(a.data(), w, h, 100, 100, 100);
    fillRGBA(b.data(), w, h, 100, 100, 100);

    setPixel(b.data(), w, 1, 1, 110, 100, 100);

    pixelmatch::Options opts;
    opts.threshold = 0.01f;  // very strict
    EXPECT_GE(pixelmatch::compare(a.data(), b.data(), w, h, nullptr, opts), 1);
}

TEST(PixelmatchTest, DiffOutput_RedPixelsAtMismatches) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4), diff(w * h * 4, 0);
    fillRGBA(a.data(), w, h, 0, 0, 0);
    fillRGBA(b.data(), w, h, 0, 0, 0);

    setPixel(b.data(), w, 2, 2, 255, 255, 255);

    pixelmatch::compare(a.data(), b.data(), w, h, diff.data());

    // Check that (2,2) is red in the diff
    int i = (2 * w + 2) * 4;
    EXPECT_EQ(diff[i + 0], 255) << "Diff pixel R should be 255";
    EXPECT_EQ(diff[i + 1], 0)   << "Diff pixel G should be 0";
    EXPECT_EQ(diff[i + 2], 0)   << "Diff pixel B should be 0";
    EXPECT_EQ(diff[i + 3], 255) << "Diff pixel A should be 255";
}

TEST(PixelmatchTest, DiffMask_OnlyShowsMismatches) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4), diff(w * h * 4, 0);
    fillRGBA(a.data(), w, h, 128, 128, 128);
    fillRGBA(b.data(), w, h, 128, 128, 128);

    setPixel(b.data(), w, 0, 0, 255, 0, 0);

    pixelmatch::Options opts;
    opts.diffMask = true;
    pixelmatch::compare(a.data(), b.data(), w, h, diff.data(), opts);

    // Non-mismatched pixel should be black (0,0,0,0)
    int i = (1 * w + 1) * 4;
    EXPECT_EQ(diff[i], 0);
    EXPECT_EQ(diff[i + 1], 0);
    EXPECT_EQ(diff[i + 2], 0);
    EXPECT_EQ(diff[i + 3], 0);
}

TEST(PixelmatchTest, ComparePercent_ReturnsCorrectPercentage) {
    const int w = 10, h = 10;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4);
    fillRGBA(a.data(), w, h, 0, 0, 0);
    fillRGBA(b.data(), w, h, 0, 0, 0);

    // Change 10 pixels (10% of 100)
    for (int i = 0; i < 10; i++) setPixel(b.data(), w, i, 0, 255, 255, 255);

    float pct = pixelmatch::comparePercent(a.data(), b.data(), w, h);
    EXPECT_NEAR(pct, 10.0f, 0.1f);
}

TEST(PixelmatchTest, AlphaDifference_DetectedCorrectly) {
    const int w = 4, h = 4;
    std::vector<uint8_t> a(w * h * 4), b(w * h * 4);
    fillRGBA(a.data(), w, h, 200, 200, 200, 255);
    fillRGBA(b.data(), w, h, 200, 200, 200, 0);

    int mismatches = pixelmatch::compare(a.data(), b.data(), w, h);
    EXPECT_GT(mismatches, 0) << "Alpha 255 vs 0 should be detected";
}

TEST(PixelmatchTest, EmptyImage_ZeroMismatches) {
    EXPECT_EQ(pixelmatch::compare(nullptr, nullptr, 0, 0), 0);
}

TEST(PixelmatchTest, ColorDelta_YIQComponents) {
    // Pure red pixel vs pure blue pixel — large perceptual difference
    uint8_t a[] = {255, 0, 0, 255};
    uint8_t b[] = {0, 0, 255, 255};
    float delta = pixelmatch::detail::colorDelta(a, b, 0, 0, false);
    EXPECT_NE(delta, 0.0f) << "Red vs blue should have non-zero delta";
}

TEST(PixelmatchTest, ColorDelta_IdenticalPixels_ZeroDelta) {
    uint8_t a[] = {42, 128, 200, 255};
    float delta = pixelmatch::detail::colorDelta(a, a, 0, 0, false);
    EXPECT_EQ(delta, 0.0f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
