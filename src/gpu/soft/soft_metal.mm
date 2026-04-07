/*
 * Metal display bridge for the software GPU renderer.
 * Creates a shared Metal texture and uploads VRAM data to it.
 */

#ifdef PCSX_METAL

#import <Metal/Metal.h>
#include <cstdint>
#include <vector>

#include "gpu/soft/soft_metal.h"

static constexpr int VRAM_W = 1024;
static constexpr int VRAM_H = 512;

struct SoftMetalDisplay {
    id<MTLTexture> texture;
};

SoftMetalDisplay *softMetalCreateDisplay(int width, int height) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    auto *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                 width:width
                                                                height:height
                                                             mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;

    auto *display = new SoftMetalDisplay();
    display->texture = [device newTextureWithDescriptor:td];
    return display;
}

void softMetalDestroyDisplay(SoftMetalDisplay *display) {
    if (!display) return;
    display->texture = nil;
    delete display;
}

void softMetalUpload16(SoftMetalDisplay *display, const uint16_t *vram16) {
    if (!display || !display->texture) return;

    static std::vector<uint32_t> rgba(VRAM_W * VRAM_H);
    for (int i = 0; i < VRAM_W * VRAM_H; i++) {
        uint16_t px = vram16[i];
        uint8_t r = ((px >> 0) & 0x1f) * 255 / 31;
        uint8_t g = ((px >> 5) & 0x1f) * 255 / 31;
        uint8_t b = ((px >> 10) & 0x1f) * 255 / 31;
        rgba[i] = r | (g << 8) | (b << 16) | 0xff000000u;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, VRAM_W, VRAM_H);
    [display->texture replaceRegion:region mipmapLevel:0 withBytes:rgba.data() bytesPerRow:VRAM_W * 4];
}

void softMetalUpload24(SoftMetalDisplay *display, const uint8_t *vram, int byteOffset) {
    if (!display || !display->texture) return;

    // 24-bit: 3 bytes per pixel, VRAM row = 2048 bytes (1024 * 2)
    // Width in pixels = 682 (1024 * 2 / 3)
    static constexpr int W24 = 682;
    static std::vector<uint32_t> rgba(W24 * VRAM_H);

    const uint8_t *src = vram + byteOffset;
    for (int y = 0; y < VRAM_H; y++) {
        for (int x = 0; x < W24; x++) {
            int srcIdx = (y * 2048) + (x * 3);
            uint8_t r = src[srcIdx + 0];
            uint8_t g = src[srcIdx + 1];
            uint8_t b = src[srcIdx + 2];
            rgba[y * W24 + x] = r | (g << 8) | (b << 16) | 0xff000000u;
        }
    }

    // Upload to left portion of texture (682 of 1024 wide)
    MTLRegion region = MTLRegionMake2D(0, 0, W24, VRAM_H);
    [display->texture replaceRegion:region mipmapLevel:0 withBytes:rgba.data() bytesPerRow:W24 * 4];
}

void *softMetalGetTextureID(SoftMetalDisplay *display) {
    if (!display) return nullptr;
    return (__bridge void *)display->texture;
}

#endif  // PCSX_METAL
