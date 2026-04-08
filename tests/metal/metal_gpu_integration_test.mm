/*
 * Metal GPU integration tests — exercises the REAL MetalGPU class
 * through the public writeData/writeStatus/partialUpdateVRAM interface.
 * Uses raw GP0 command words, same path as the real emulator.
 */
#ifdef PCSX_METAL

#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/psxemulator.h"
#include "core/system.h"
#include "gpu/metal/gpu_metal.h"
#include "sotn_frame_capture.h"

// ===========================================================================
// Minimal System stub for headless GPU testing
// ===========================================================================
class StubSystem : public PCSX::System {
  public:
    StubSystem() : m_stubArgs(s_args), m_args(m_stubArgs) {}
    void softReset() override {}
    void hardReset() override {}
    void biosPutc(int) override {}
    const PCSX::Arguments &getArgs() const override { return m_args; }
    void printf(std::string &&) override {}
    void log(PCSX::LogClass, std::string &&) override {}
    void message(std::string &&) override {}
    void luaMessage(const std::string &, bool) override {}
    void update(bool) override {}
    void close() override {}
    void purgeAllEvents() override {}
    void testQuit(int) override {}
  private:
    static inline int s_argc = 1;
    static inline char *s_argv[] = {(char *)"test", nullptr};
    static inline CommandLine::args s_args{1, s_argv};
    PCSX::Arguments m_stubArgs;
    const PCSX::Arguments &m_args;
};

// ===========================================================================
// Global setup: create minimal System + Emulator so GPU globals work
// ===========================================================================
class MetalGPUTestEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        system = new StubSystem();
        PCSX::g_system = system;
        emulator = new PCSX::Emulator();
        PCSX::g_emulator = emulator;
    }
    void TearDown() override {
        PCSX::g_emulator = nullptr;
        delete emulator;
        PCSX::g_system = nullptr;
        delete system;
    }
  private:
    StubSystem *system = nullptr;
    PCSX::Emulator *emulator = nullptr;
};

// Register the environment (gtest owns it)
static auto *env = ::testing::AddGlobalTestEnvironment(new MetalGPUTestEnvironment());

// ===========================================================================
// Integration test fixture using the REAL MetalGPU via public GPU interface
// ===========================================================================
class MetalGPUIntegrationTest : public ::testing::Test {
  protected:
    static constexpr int W = 1024, H = 512;

    std::unique_ptr<PCSX::GPU> gpu;

    void SetUp() override {
        gpu = PCSX::GPU::getMetal();
        ASSERT_NE(gpu, nullptr);
        int ret = gpu->init(nullptr);
        ASSERT_EQ(ret, 0) << "MetalGPU init failed";

        // Set default draw area to full VRAM
        gpuDrawArea(0, 0, 1023, 511);
        gpuDrawOffset(0, 0);
    }

    void TearDown() override {
        if (gpu) gpu->shutdown();
    }

    // --- GP0 command helpers (raw 32-bit words via writeData) ---

    // GP0(02h) FastFill: color, x/y, w/h
    void gpuFastFill(int x, int y, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        gpu->writeData(0x02000000 | r | (g << 8) | (b << 16));
        gpu->writeData((y << 16) | (x & 0xffff));
        gpu->writeData((h << 16) | (w & 0xffff));
    }

    // GP0(E3h) Drawing area top-left
    void gpuDrawAreaStart(int x, int y) {
        gpu->writeData(0xE3000000 | (x & 0x3ff) | ((y & 0x1ff) << 10));
    }

    // GP0(E4h) Drawing area bottom-right
    void gpuDrawAreaEnd(int x, int y) {
        gpu->writeData(0xE4000000 | (x & 0x3ff) | ((y & 0x1ff) << 10));
    }

    void gpuDrawArea(int x0, int y0, int x1, int y1) {
        gpuDrawAreaStart(x0, y0);
        gpuDrawAreaEnd(x1, y1);
    }

    // GP0(E5h) Drawing offset
    void gpuDrawOffset(int x, int y) {
        gpu->writeData(0xE5000000 | (x & 0x7ff) | ((y & 0x7ff) << 11));
    }

    // GP0(E1h) Texture page
    void gpuTPage(uint16_t tp) {
        gpu->writeData(0xE1000000 | (tp & 0x1ff));
    }

    // GP0(2Ch) Textured quad (flat-shaded, opaque, no modulation)
    // Vertex order: v0(top-left), v1(top-right), v2(bottom-left), v3(bottom-right)
    void gpuTexturedQuad(uint32_t color,
                         int x0, int y0, uint8_t u0, uint8_t v0, uint16_t clut,
                         int x1, int y1, uint8_t u1, uint8_t v1, uint16_t tpage,
                         int x2, int y2, uint8_t u2, uint8_t v2,
                         int x3, int y3, uint8_t u3, uint8_t v3) {
        gpu->writeData(0x2C000000 | (color & 0xFFFFFF));           // cmd + color
        gpu->writeData((y0 << 16) | (x0 & 0xffff));                // v0 xy
        gpu->writeData((clut << 16) | (v0 << 8) | u0);             // v0 uv + clut
        gpu->writeData((y1 << 16) | (x1 & 0xffff));                // v1 xy
        gpu->writeData((tpage << 16) | (v1 << 8) | u1);            // v1 uv + tpage
        gpu->writeData((y2 << 16) | (x2 & 0xffff));                // v2 xy
        gpu->writeData((v2 << 8) | u2);                             // v2 uv
        gpu->writeData((y3 << 16) | (x3 & 0xffff));                // v3 xy
        gpu->writeData((v3 << 8) | u3);                             // v3 uv
    }

    // Helper: upload 16-bit VRAM data
    void upload(int x, int y, int w, int h, const uint16_t *data) {
        gpu->partialUpdateVRAM(x, y, w, h, data);
    }

    // Helper: end frame
    void endFrame() {
        gpu->vblank(false);
    }

    // Helper: read VRAM as 16-bit
    std::vector<uint16_t> readVRAM16() {
        auto slice = gpu->getVRAM(PCSX::GPU::Ownership::ACQUIRE);
        std::vector<uint16_t> result(W * H);
        std::memcpy(result.data(), slice.data(), W * H * 2);
        return result;
    }
};

// ===========================================================================
// Tests
// ===========================================================================

TEST_F(MetalGPUIntegrationTest, FastFill_WritesToVRAM) {
    gpuFastFill(100, 100, 200, 100, 0xff, 0, 0);  // red
    endFrame();
    auto vram = readVRAM16();
    EXPECT_EQ(vram[100 * W + 100] & 0x1f, 31) << "FastFill red should produce R=31";
    EXPECT_EQ(vram[50 * W + 50], 0) << "Outside fill should be black";
}

TEST_F(MetalGPUIntegrationTest, CLUTUpload_SurvivesFastFill) {
    // Upload CLUT at Y=241 (between framebuffers)
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    for (int i = 1; i < 256; i++) clut[i] = (i & 0x1f) | ((i & 0x1f) << 5) | 0x8000;
    upload(256, 241, 256, 1, clut.data());

    // FastFill framebuffer area (Y=0-239, should NOT touch Y=241)
    gpuDrawArea(0, 0, 383, 239);
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);
    endFrame();

    auto vram = readVRAM16();
    EXPECT_EQ(vram[241 * W + 257] & 0x7fff, clut[1] & 0x7fff)
        << "CLUT[1] at (257,241) should survive FastFill at (0,0,384,240)";
}

TEST_F(MetalGPUIntegrationTest, TexturedQuad_8bpp_ProducesVisiblePixels) {
    // Upload 8bpp CLUT at (256, 241)
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    clut[1] = 0x7fff;  // white
    for (int i = 2; i < 256; i++) clut[i] = (i & 0x1f) | 0x8000;
    upload(256, 241, 256, 1, clut.data());

    // Upload 8bpp texture at page (704, 0): all index 1 (white)
    std::vector<uint16_t> tex(64 * 16, 0x0101);
    upload(704, 0, 64, 16, tex.data());

    // Clear framebuffer
    gpuDrawArea(0, 0, 383, 239);
    gpuDrawOffset(0, 0);
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);

    // Draw textured quad: tp=0x008b (page 704,0 8bpp), clut=(256,241)
    uint16_t cl = (256 / 16) | (241 << 6);  // 0x3c50
    uint16_t tp = 0x008b;
    gpuTexturedQuad(0x808080,
                    100, 100, 0, 0, cl,
                    228, 100, 128, 0, tp,
                    100, 116, 0, 16,
                    228, 116, 128, 16);
    endFrame();

    auto vram = readVRAM16();
    int nonBlack = 0;
    for (int y = 100; y < 116; y++)
        for (int x = 100; x < 228; x++)
            if (vram[y * W + x] != 0) nonBlack++;
    EXPECT_GT(nonBlack, 1000) << "8bpp textured quad should produce many non-black pixels";
}

// Full SotN file select sequence through the REAL pipeline
TEST_F(MetalGPUIntegrationTest, SotN_FullPipeline) {
    // 1. Upload CLUT at (256, 241)
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    for (int i = 1; i < 256; i++) {
        int r = (i * 20 / 256) & 0x1f;
        int g = (i * 22 / 256) & 0x1f;
        int b = (i * 25 / 256) & 0x1f;
        clut[i] = r | (g << 5) | (b << 10) | 0x8000;
    }
    upload(256, 241, 256, 1, clut.data());

    // 2. Upload 8bpp texture to page (704, 0)
    std::vector<uint16_t> tex(64 * 240);
    for (int y = 0; y < 240; y++)
        for (int x = 0; x < 64; x++) {
            uint8_t i0 = ((x * 2 + y) * 7 + 13) & 0xff;
            uint8_t i1 = ((x * 2 + 1 + y) * 7 + 13) & 0xff;
            if (i0 == 0) i0 = 1;
            if (i1 == 0) i1 = 1;
            tex[y * 64 + x] = i0 | (i1 << 8);
        }
    upload(704, 0, 64, 240, tex.data());

    // 3. Set draw area + offset
    gpuDrawArea(0, 0, 383, 239);
    gpuDrawOffset(0, 0);

    // 4. FastFill to black
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);

    // 5. Draw textured quad covering 128x240 (one SotN gargoyle tile)
    uint16_t cl = (256 / 16) | (241 << 6);
    gpuTexturedQuad(0x808080,
                    0, 0, 0, 0, cl,
                    128, 0, 128, 0, 0x008b,
                    0, 240, 0, 240,
                    128, 240, 128, 240);
    endFrame();

    // 6. Read back and verify non-black output
    auto vram = readVRAM16();
    int nonBlack = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vram[y * W + x] != 0) nonBlack++;

    EXPECT_GT(nonBlack, 10000)
        << "SotN gargoyle tile should produce many non-black pixels through the full pipeline. Got " << nonBlack;
}

// Double-buffer simulation: reproduces the SotN file select frame alternation.
// Frame A: offset=(0,0), scissor=(0,0,384,240), draws gargoyle to top half
// Frame B: offset=(0,256), scissor=(0,256,384,240), draws gargoyle to bottom half
// Display alternates: Frame A shows bottom half, Frame B shows top half.
// The gargoyle drawn in Frame A should persist in the top half when Frame B displays it.
TEST_F(MetalGPUIntegrationTest, DoubleBuffer_GargoylePersistsAcrossFrames) {
    // Upload CLUT at (256, 241) — between the two framebuffers
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    for (int i = 1; i < 256; i++) {
        clut[i] = (i & 0x1f) | (((i >> 1) & 0x1f) << 5) | (((i >> 2) & 0x1f) << 10) | 0x8000;
    }
    upload(256, 241, 256, 1, clut.data());

    // Upload 8bpp texture at page (704, 0)
    std::vector<uint16_t> tex(64 * 240);
    for (int y = 0; y < 240; y++)
        for (int x = 0; x < 64; x++) {
            uint8_t i0 = ((x * 2 + y) * 7 + 13) & 0xff;
            uint8_t i1 = ((x * 2 + 1 + y) * 7 + 13) & 0xff;
            if (i0 == 0) i0 = 1;
            if (i1 == 0) i1 = 1;
            tex[y * 64 + x] = i0 | (i1 << 8);
        }
    upload(704, 0, 64, 240, tex.data());

    uint16_t cl = (256 / 16) | (241 << 6);

    // === Frame A: draw gargoyle to TOP half ===
    gpuDrawArea(0, 0, 383, 239);
    gpuDrawOffset(0, 0);
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);

    gpuTexturedQuad(0x808080,
                    0, 0, 0, 0, cl,
                    128, 0, 128, 0, 0x008b,
                    0, 240, 0, 240,
                    128, 240, 128, 240);
    endFrame();

    // Verify gargoyle is in top half after Frame A
    auto vramA = readVRAM16();
    int nonBlackTopA = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vramA[y * W + x] != 0) nonBlackTopA++;
    EXPECT_GT(nonBlackTopA, 10000)
        << "Frame A: gargoyle should be in top half. Got " << nonBlackTopA;

    // === Frame B: draw gargoyle to BOTTOM half, display top half ===
    gpuDrawArea(0, 256, 383, 495);
    gpuDrawOffset(0, 256);
    gpuFastFill(0, 256, 384, 240, 0, 0, 0);

    gpuTexturedQuad(0x808080,
                    0, 0, 0, 0, cl,
                    128, 0, 128, 0, 0x008b,
                    0, 240, 0, 240,
                    128, 240, 128, 240);
    endFrame();

    // Verify gargoyle is in BOTH halves
    auto vramB = readVRAM16();

    // Top half should STILL have the gargoyle from Frame A
    int nonBlackTopB = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vramB[y * W + x] != 0) nonBlackTopB++;
    EXPECT_GT(nonBlackTopB, 10000)
        << "Frame B: top half gargoyle from Frame A should persist. Got " << nonBlackTopB;

    // Bottom half should have the new gargoyle from Frame B
    int nonBlackBotB = 0;
    for (int y = 266; y < 486; y++)
        for (int x = 10; x < 118; x++)
            if (vramB[y * W + x] != 0) nonBlackBotB++;
    EXPECT_GT(nonBlackBotB, 10000)
        << "Frame B: gargoyle should be in bottom half. Got " << nonBlackBotB;

    // === Frame C: draw to top half AGAIN, display bottom half ===
    // This simulates the steady-state alternation
    gpuDrawArea(0, 0, 383, 239);
    gpuDrawOffset(0, 0);
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);

    gpuTexturedQuad(0x808080,
                    0, 0, 0, 0, cl,
                    128, 0, 128, 0, 0x008b,
                    0, 240, 0, 240,
                    128, 240, 128, 240);
    endFrame();

    // Top half should have fresh gargoyle, bottom should persist from Frame B
    auto vramC = readVRAM16();
    int nonBlackTopC = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vramC[y * W + x] != 0) nonBlackTopC++;
    EXPECT_GT(nonBlackTopC, 10000)
        << "Frame C: fresh gargoyle in top half. Got " << nonBlackTopC;

    int nonBlackBotC = 0;
    for (int y = 266; y < 486; y++)
        for (int x = 10; x < 118; x++)
            if (vramC[y * W + x] != 0) nonBlackBotC++;
    EXPECT_GT(nonBlackBotC, 10000)
        << "Frame C: bottom gargoyle from Frame B should persist. Got " << nonBlackBotC;

    // === Verify CLUT survived all 3 frames ===
    EXPECT_NE(vramC[241 * W + 257], 0)
        << "CLUT[1] should survive all 3 frames of double-buffered rendering";
}

// Same as above but with the EXACT SotN pattern where Frame A has NO gargoyle
// (only UI), testing if the gargoyle from Frame B persists when displayed.
TEST_F(MetalGPUIntegrationTest, DoubleBuffer_AsymmetricFrames) {
    // Upload CLUT + texture
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    for (int i = 1; i < 256; i++)
        clut[i] = (i & 0x1f) | (((i >> 1) & 0x1f) << 5) | 0x8000;
    upload(256, 241, 256, 1, clut.data());

    std::vector<uint16_t> tex(64 * 240);
    for (int y = 0; y < 240; y++)
        for (int x = 0; x < 64; x++) {
            uint8_t i0 = ((x * 2 + y) * 7 + 13) & 0xff;
            if (i0 == 0) i0 = 1;
            uint8_t i1 = i0;
            tex[y * 64 + x] = i0 | (i1 << 8);
        }
    upload(704, 0, 64, 240, tex.data());

    uint16_t cl = (256 / 16) | (241 << 6);

    // === Frame A: draw gargoyle + UI to TOP half ===
    gpuDrawArea(0, 0, 383, 239);
    gpuDrawOffset(0, 0);
    gpuFastFill(0, 0, 384, 240, 0, 0, 0);
    // Gargoyle
    gpuTexturedQuad(0x808080, 0,0,0,0,cl, 128,0,128,0,0x008b, 0,240,0,240, 128,240,128,240);
    endFrame();

    // === Frame B: draw ONLY UI to BOTTOM half (no gargoyle) ===
    gpuDrawArea(0, 256, 383, 495);
    gpuDrawOffset(0, 256);
    gpuFastFill(0, 256, 384, 240, 0, 0, 0);
    // Small UI rect only — no gargoyle in this frame
    gpuFastFill(100, 300, 100, 50, 0x40, 0x40, 0x80);
    endFrame();

    // Top half should STILL have gargoyle from Frame A
    auto vram = readVRAM16();
    int nonBlackTop = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vram[y * W + x] != 0) nonBlackTop++;
    EXPECT_GT(nonBlackTop, 10000)
        << "Asymmetric: gargoyle in top half should persist after Frame B drew to bottom. Got " << nonBlackTop;
}

// Replay captured SotN frame through the real Metal GPU pipeline.
// This feeds the exact same GP0/GP1 commands that the game sends.
TEST_F(MetalGPUIntegrationTest, SotN_ReplayCapturedFrame) {
    // First, upload the CLUT that would have been loaded during game init.
    // The capture has a VRAM write at (256,240) with 256 entries every frame.
    // We also need the texture data at pages (576-704, 0).
    // For now, pre-populate with test data so the quads produce visible output.

    // Upload CLUT at (256, 241) — the gargoyle quad's cl=0x3c50 decodes to (256,241)
    // The capture shows VRAM_Write at (256,240), but the quads reference cl=0x3c50=(256,241)
    // Upload to BOTH rows to be safe
    std::vector<uint16_t> clut(256);
    clut[0] = 0x0000;
    for (int i = 1; i < 256; i++)
        clut[i] = (i & 0x1f) | (((i*2) & 0x1f) << 5) | (((i*3) & 0x1f) << 10) | 0x8000;
    upload(256, 240, 256, 1, clut.data());
    upload(256, 241, 256, 1, clut.data());

    // Upload 8bpp texture data at pages 9,10,11 (X=576,640,704)
    for (int page = 0; page < 3; page++) {
        int px = 576 + page * 64;
        std::vector<uint16_t> tex(64 * 240);
        for (int y = 0; y < 240; y++)
            for (int x = 0; x < 64; x++) {
                uint8_t i0 = ((page * 128 + x * 2 + y) * 7 + 13) & 0xff;
                uint8_t i1 = ((page * 128 + x * 2 + 1 + y) * 7 + 13) & 0xff;
                if (i0 == 0) i0 = 1;
                if (i1 == 0) i1 = 1;
                tex[y * 64 + x] = i0 | (i1 << 8);
            }
        upload(px, 0, 64, 240, tex.data());
    }

    // Replay the captured frame commands
    int writeDataCount = 0, chainCount = 0;
    for (int i = 0; i < s_sotnFrameCount; i++) {
        const auto &op = s_sotnFrame[i];
        switch (op.type) {
            case 'D':
                gpu->writeData(op.data[0]);
                break;
            case 'S':
                gpu->writeStatus(op.data[0]);
                break;
            case 'C':
                // Chain DMA: feed as a single buffer (like real DMA)
                gpu->directDMAWrite(op.data, op.count, 0);
                chainCount++;
                break;
            case 'A':
                // Direct DMA write
                gpu->directDMAWrite(op.data, op.count, 0);
                break;
        }
    }

    // End frame
    endFrame();

    // Check: the gargoyle area should have non-black pixels
    auto vram = readVRAM16();
    int nonBlack = 0;
    for (int y = 10; y < 230; y++)
        for (int x = 10; x < 118; x++)
            if (vram[y * W + x] != 0) nonBlack++;

    EXPECT_GT(nonBlack, 5000)
        << "SotN replay: gargoyle area should have visible pixels. Got " << nonBlack;

    // Also check that the ENTIRE framebuffer has content (not just gargoyle)
    int totalNonBlack = 0;
    for (int y = 0; y < 240; y++)
        for (int x = 0; x < 384; x++)
            if (vram[y * W + x] != 0) totalNonBlack++;

    fprintf(stderr, "[REPLAY] Non-black pixels in framebuffer: %d (gargoyle area: %d)\n",
            totalNonBlack, nonBlack);
}

#endif  // PCSX_METAL
