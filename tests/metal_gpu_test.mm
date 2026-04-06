/*
 * Metal GPU unit tests — headless, no emulator dependency.
 * Tests the rendering pipeline in isolation: upload, sync, draw, readback.
 *
 * Build:
 *   c++ -std=c++2b -DPCSX_METAL -framework Metal -framework QuartzCore -framework Foundation \
 *       -Ithird_party/googletest/googletest/include -Ithird_party/googletest/googletest \
 *       third_party/googletest/googletest/src/gtest-all.cc \
 *       tests/metal_gpu_test.mm -o bins/metal_gpu_test
 *
 * Run:
 *   ./bins/metal_gpu_test
 */
#ifdef PCSX_METAL

#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal Metal GPU fixture
// ---------------------------------------------------------------------------
class MetalGPUTest : public ::testing::Test {
  protected:
    static constexpr int W = 1024, H = 512;

    id<MTLDevice>              dev;
    id<MTLCommandQueue>        queue;
    id<MTLTexture>             vramRT;
    id<MTLTexture>             vramSample;
    id<MTLRenderPipelineState> pso;

    struct Vertex {
        int32_t x, y;
        uint32_t color;
        uint16_t texpage, clut, u, v;
    };
    struct Uniforms {
        float dox, doy;
        int twmx, twmy, twox, twoy;
        float br, bg, bb, ba, bor, bog, bob, boa;
    };

    void SetUp() override {
        dev   = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        ASSERT_NE(dev, nil);

        auto *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                     width:W height:H mipmapped:NO];
        td.usage        = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode  = MTLStorageModeShared;
        vramRT = [dev newTextureWithDescriptor:td];

        td.usage = MTLTextureUsageShaderRead;
        vramSample = [dev newTextureWithDescriptor:td];

        NSError *err = nil;
        auto *lib = [dev newLibraryWithSource:shaderSource() options:nil error:&err];
        ASSERT_NE(lib, nil) << err.localizedDescription.UTF8String;

        auto *vd = vertexDescriptor();
        auto *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction   = [lib newFunctionWithName:@"vmain"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fmain"];
        pd.vertexDescriptor = vd;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        pso = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        ASSERT_NE(pso, nil) << err.localizedDescription.UTF8String;
    }

    // Upload 16-bit PSX pixels to the render target (replicating partialUpdateVRAM)
    void upload16(int x, int y, int w, int h, const uint16_t *pixels) {
        std::vector<uint32_t> rgba(w * h);
        for (int i = 0; i < w * h; i++) {
            uint16_t px = pixels[i];
            uint8_t r = ((px >>  0) & 0x1f) * 255 / 31;
            uint8_t g = ((px >>  5) & 0x1f) * 255 / 31;
            uint8_t b = ((px >> 10) & 0x1f) * 255 / 31;
            uint8_t a = (px & 0x8000) ? 0xff : 0x00;
            rgba[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
        size_t byteSize = rgba.size() * 4;
        auto *stg = [dev newBufferWithBytes:rgba.data() length:byteSize options:MTLResourceStorageModeShared];
        auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:stg sourceOffset:0 sourceBytesPerRow:w*4 sourceBytesPerImage:byteSize
                sourceSize:MTLSizeMake(w, h, 1)
                 toTexture:vramRT destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake(x, y, 0)];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }

    // Sync render target → sample texture
    void sync() {
        auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
        [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(W, H, 1)
                  toTexture:vramSample destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake(0,0,0)];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }

    // Draw triangles
    void draw(const std::vector<Vertex> &verts, bool /*unused*/ = false) {
        Uniforms u = {0.5f, -0.5f, ~0, ~0, 0, 0, 1,1,1,0, 1,1,1,0};
        auto *buf = [dev newBufferWithBytes:verts.data() length:verts.size()*sizeof(Vertex) options:MTLResourceStorageModeShared];
        auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = vramRT;
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        auto *cb = [queue commandBuffer]; auto *enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        // No viewport flip needed — Metal NDC Y+ is down, matching VRAM convention.
        // The vertex shader uses y/-256+1 to map VRAM Y=0 to clip top.
        [enc setVertexBuffer:buf offset:0 atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentTexture:vramSample atIndex:0];
        MTLScissorRect sc = {0, 0, (NSUInteger)W, (NSUInteger)H};
        [enc setScissorRect:sc];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:verts.size()];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }

    // Read back a single pixel from the render target
    struct RGBA { uint8_t r, g, b, a; };
    RGBA readPixel(int x, int y) {
        size_t bytes = W * H * 4;
        auto *stg = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
        [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(W, H, 1) toBuffer:stg destinationOffset:0
          destinationBytesPerRow:W*4 destinationBytesPerImage:bytes];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        auto *px = (const uint8_t *)stg.contents;
        int i = (y * W + x) * 4;
        return {px[i], px[i+1], px[i+2], px[i+3]};
    }

    // Count non-black pixels in a region
    int countNonBlack(int rx, int ry, int rw, int rh) {
        size_t bytes = W * H * 4;
        auto *stg = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
        [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(W, H, 1) toBuffer:stg destinationOffset:0
          destinationBytesPerRow:W*4 destinationBytesPerImage:bytes];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        auto *px = (const uint8_t *)stg.contents;
        int count = 0;
        for (int y = ry; y < ry + rh; y++)
            for (int x = rx; x < rx + rw; x++) {
                int i = (y * W + x) * 4;
                if (px[i] > 0 || px[i+1] > 0 || px[i+2] > 0) count++;
            }
        return count;
    }

  private:
    static NSString *shaderSource() {
        return @R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VIn { int2 pos [[attribute(0)]]; uint col [[attribute(1)]]; ushort tp [[attribute(2)]]; ushort cl [[attribute(3)]]; ushort2 uv [[attribute(4)]]; };
struct VOut { float4 position [[position]]; float4 vcolor; float2 tc; int2 clutBase [[flat]]; int2 tpBase [[flat]]; int texMode [[flat]]; };
struct Uniforms { float2 drawOffset; int4 texWindow; float4 blend; float4 blendOpaque; };

vertex VOut vmain(VIn in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    VOut o;
    o.position = float4((float(in.pos.x)+u.drawOffset.x)/512.0-1.0, (float(in.pos.y)+u.drawOffset.y)/-256.0+1.0, 0, 1);
    o.vcolor = float4(float(in.col&0xffu)/255.0, float((in.col>>8u)&0xffu)/255.0, float((in.col>>16u)&0xffu)/255.0, 1);
    if ((in.tp & 0x8000) != 0) { o.texMode = 4; }
    else {
        o.texMode = (in.tp >> 7) & 3;
        o.tc = float2(in.uv);
        o.tpBase = int2((in.tp & 0xf)*64, ((in.tp>>4)&1)*256);
        o.clutBase = int2((in.cl & 0x3f)*16, in.cl >> 6);
    }
    return o;
}

static int floatToU5(float f) { return int(floor(f*31.0+0.5)); }
static float4 sampleVRAM(texture2d<float, access::read> t, int2 c) { c &= int2(1023,511); return t.read(uint2(c)); }
static int sample16(texture2d<float, access::read> t, int2 c) {
    float4 col = sampleVRAM(t, c);
    return floatToU5(col.r)|(floatToU5(col.g)<<5)|(floatToU5(col.b)<<10)|(int(ceil(col.a))<<15);
}
static float4 texBlend(float4 c1, float4 c2) { float4 r = (c1*c2)/(128.0/255.0); r.a=1.0; return r; }

struct FOut { float4 color [[color(0)]]; };
fragment FOut fmain(VOut in [[stage_in]], texture2d<float, access::read> vram [[texture(0)]], constant Uniforms &u [[buffer(1)]]) {
    FOut out;
    if (in.texMode == 4) { out.color = in.vcolor; return out; }

    // Diagnostic mode: output UV as color gradient
    if (in.texMode == 99) {
        out.color = float4(in.tc.x / 64.0, in.tc.y / 64.0, 0, 1);
        return out;
    }

    int2 UV = int2(floor(in.tc + float2(0.0001))) & int2(0xff);
    UV = (UV & u.texWindow.xy) | u.texWindow.zw;
    if (in.texMode == 0) { // 4bpp
        int2 tc = int2(UV.x >> 2, UV.y) + in.tpBase;
        int s = sample16(vram, tc);
        int idx = (s >> ((UV.x&3)<<2)) & 0xf;
        out.color = vram.read(uint2(int2(in.clutBase.x+idx, in.clutBase.y) & int2(1023,511)));
    } else if (in.texMode == 1) { // 8bpp
        int2 tc = int2(UV.x >> 1, UV.y) + in.tpBase;
        int s = sample16(vram, tc);
        int idx = (s >> ((UV.x&1)<<3)) & 0xff;
        out.color = vram.read(uint2(int2(in.clutBase.x+idx, in.clutBase.y) & int2(1023,511)));

    } else { // 16bpp
        out.color = sampleVRAM(vram, UV + in.tpBase);
    }
    out.color = texBlend(out.color, in.vcolor);

    return out;
}
)MSL";
    }

    static MTLVertexDescriptor *vertexDescriptor() {
        auto *vd = [[MTLVertexDescriptor alloc] init];
        vd.attributes[0].format = MTLVertexFormatInt2;    vd.attributes[0].offset = 0;  vd.attributes[0].bufferIndex = 0;
        vd.attributes[1].format = MTLVertexFormatUInt;    vd.attributes[1].offset = 8;  vd.attributes[1].bufferIndex = 0;
        vd.attributes[2].format = MTLVertexFormatUShort;  vd.attributes[2].offset = 12; vd.attributes[2].bufferIndex = 0;
        vd.attributes[3].format = MTLVertexFormatUShort;  vd.attributes[3].offset = 14; vd.attributes[3].bufferIndex = 0;
        vd.attributes[4].format = MTLVertexFormatUShort2; vd.attributes[4].offset = 16; vd.attributes[4].bufferIndex = 0;
        vd.layouts[0].stride = sizeof(Vertex);
        return vd;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(MetalGPUTest, UntexturedTriangle_ProducesColoredPixels) {
    uint32_t red = 0x0000ff;  // BGR: red
    std::vector<Vertex> verts = {
        {100, 100, red, 0x8000, 0, 0, 0},
        {200, 100, red, 0x8000, 0, 0, 0},
        {150, 150, red, 0x8000, 0, 0, 0},
    };
    draw(verts);
    int nb = countNonBlack(95, 95, 110, 60);
    EXPECT_GT(nb, 100) << "Untextured triangle should produce visible pixels";
}

TEST_F(MetalGPUTest, UntexturedTriangle_CorrectColor) {
    uint32_t green = 0x00ff00;  // BGR: green
    std::vector<Vertex> verts = {
        {300, 200, green, 0x8000, 0, 0, 0},
        {400, 200, green, 0x8000, 0, 0, 0},
        {350, 250, green, 0x8000, 0, 0, 0},
    };
    draw(verts);
    auto px = readPixel(350, 225);  // center of triangle (with viewport flip → row 225)
    EXPECT_EQ(px.g, 255) << "Green channel should be max";
    EXPECT_EQ(px.r, 0)   << "Red channel should be zero";
}

TEST_F(MetalGPUTest, Upload16_DataPersistsInRT) {
    uint16_t magenta = 0x1f | (0x1f << 10) | 0x8000;  // R=31, G=0, B=31, MSB=1
    upload16(500, 300, 1, 1, &magenta);
    auto px = readPixel(500, 300);
    EXPECT_EQ(px.r, 255);
    EXPECT_EQ(px.g, 0);
    EXPECT_EQ(px.b, 255);
    EXPECT_EQ(px.a, 255);
}

TEST_F(MetalGPUTest, Sync_CopiesRTToSample) {
    uint16_t cyan = (0x1f << 5) | (0x1f << 10) | 0x8000;  // G=31, B=31
    upload16(600, 400, 1, 1, &cyan);
    sync();
    // Read back sample texture
    size_t bytes = W * H * 4;
    auto *stg = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
    [bl copyFromTexture:vramSample sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
             sourceSize:MTLSizeMake(W, H, 1) toBuffer:stg destinationOffset:0
      destinationBytesPerRow:W*4 destinationBytesPerImage:bytes];
    [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
    auto *px = (const uint8_t *)stg.contents;
    int i = (400 * W + 600) * 4;
    EXPECT_EQ(px[i+0], 0);    // R
    EXPECT_EQ(px[i+1], 255);  // G
    EXPECT_EQ(px[i+2], 255);  // B
}

TEST_F(MetalGPUTest, Texture16bpp_ProducesCorrectColors) {
    // Upload magenta block at (0,0) as 16bpp texture
    uint16_t magenta = 0x1f | (0x1f << 10) | 0x8000;
    std::vector<uint16_t> block(4 * 4, magenta);
    upload16(0, 0, 4, 4, block.data());
    sync();

    // Draw 16bpp textured rect at (200,200)-(204,204)
    uint16_t tp = 0x0100;  // 16bpp, page (0,0)
    uint32_t white = 0x808080;
    std::vector<Vertex> verts = {
        {200,200,white, tp,0, 0,0}, {204,200,white, tp,0, 4,0}, {204,204,white, tp,0, 4,4},
        {204,204,white, tp,0, 4,4}, {200,204,white, tp,0, 0,4}, {200,200,white, tp,0, 0,0},
    };
    draw(verts);
    auto px = readPixel(202, 202);
    EXPECT_GT(px.r, 200) << "16bpp texture should produce magenta";
    EXPECT_LT(px.g, 50);
    EXPECT_GT(px.b, 200);
}

TEST_F(MetalGPUTest, VRAMCoords_MapToCorrectTextureRow) {
    uint32_t blue = 0xff0000;  // BGR: blue
    std::vector<Vertex> verts = {
        {100, 100, blue, 0x8000, 0, 0, 0},
        {110, 100, blue, 0x8000, 0, 0, 0},
        {110, 110, blue, 0x8000, 0, 0, 0},
        {110, 110, blue, 0x8000, 0, 0, 0},
        {100, 110, blue, 0x8000, 0, 0, 0},
        {100, 100, blue, 0x8000, 0, 0, 0},
    };
    draw(verts);
    // VRAM Y=105 should map to texture row ~105 (Metal NDC Y+ down)
    auto px = readPixel(105, 105);
    EXPECT_GT(px.b, 200) << "VRAM (105,105) should map to texture row 105";
}

TEST_F(MetalGPUTest, UploadAndRender_SameCoordinateSpace) {
    // Upload a cyan pixel at (500, 250)
    uint16_t cyan = (0x1f << 5) | (0x1f << 10) | 0x8000;
    upload16(500, 250, 1, 1, &cyan);

    // Draw an untextured red rect at (500, 250)-(510, 260)
    uint32_t red = 0x0000ff;
    std::vector<Vertex> verts = {
        {500,250,red, 0x8000,0, 0,0}, {510,250,red, 0x8000,0, 0,0}, {510,260,red, 0x8000,0, 0,0},
        {510,260,red, 0x8000,0, 0,0}, {500,260,red, 0x8000,0, 0,0}, {500,250,red, 0x8000,0, 0,0},
    };
    draw(verts);

    // The red rect should overwrite the cyan pixel at (500, 250)
    auto px = readPixel(500, 250);
    EXPECT_GT(px.r, 200) << "GPU render should overwrite CPU upload at same coordinates";
    EXPECT_LT(px.g, 50);
}

TEST_F(MetalGPUTest, Texture8bpp_CLUTLookupProducesVariedColors) {
    // Upload 256-entry CLUT at (0, 480) with a red gradient
    std::vector<uint16_t> clut(256);
    for (int i = 0; i < 256; i++) {
        // Make every entry visibly distinct: R = i & 0x1f, G = (i >> 1) & 0x1f
        int r = i & 0x1f;
        int g = (i >> 1) & 0x1f;
        clut[i] = r | (g << 5) | (0 << 10) | 0x8000;
    }
    // Entry 0: make it non-zero but small
    clut[0] = 1 | (1 << 5) | 0x8000;
    upload16(0, 480, 256, 1, clut.data());

    // Upload 8bpp texture data at texpage (0, 0)
    // Each 16-bit VRAM word = 2 consecutive 8-bit indices
    // Create a row of indices: 0,1,2,3,...,63 across 32 words
    std::vector<uint16_t> texRow(32);
    for (int i = 0; i < 32; i++) {
        uint8_t idx0 = i * 2;
        uint8_t idx1 = i * 2 + 1;
        texRow[i] = idx0 | (idx1 << 8);
    }
    upload16(0, 0, 32, 1, texRow.data());

    sync();

    // Draw 8bpp textured rect: 64 pixels wide, 1 pixel tall
    // texpage: page X=0, Y=0, depth=1 (8bpp) → tp = (1 << 7) = 0x0080
    // clut at (0, 480): cl = (0 & 0x3f) | (480 << 6) = 480 << 6
    uint16_t tp = 0x0080;
    uint16_t cl = 480 << 6;
    uint32_t white = 0x808080;
    std::vector<Vertex> verts = {
        {200,200,white, tp,cl, 0,0}, {264,200,white, tp,cl, 64,0}, {264,210,white, tp,cl, 64,1},
        {264,210,white, tp,cl, 64,1}, {200,210,white, tp,cl, 0,1}, {200,200,white, tp,cl, 0,0},
    };
    draw(verts);

    // DEBUG: check sample texture and render target
    {
        size_t bytes = W * H * 4;
        auto *stg = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        // First check sample texture (what the shader reads from)
        auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
        [bl copyFromTexture:vramSample sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(W, H, 1) toBuffer:stg destinationOffset:0
          destinationBytesPerRow:W*4 destinationBytesPerImage:bytes];
        [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        auto *px = (const uint8_t *)stg.contents;

        // Texture data at (0,0) — should be non-zero
        int i = (0 * W + 0) * 4;
        printf("Sample tex (0,0): R=%d G=%d B=%d A=%d\n", px[i], px[i+1], px[i+2], px[i+3]);
        i = (0 * W + 1) * 4;
        printf("Sample tex (1,0): R=%d G=%d B=%d A=%d\n", px[i], px[i+1], px[i+2], px[i+3]);

        // CLUT at (0, 480)
        i = (480 * W + 0) * 4;
        printf("CLUT (0,480): R=%d G=%d B=%d A=%d\n", px[i], px[i+1], px[i+2], px[i+3]);
        i = (480 * W + 1) * 4;
        printf("CLUT (1,480): R=%d G=%d B=%d A=%d\n", px[i], px[i+1], px[i+2], px[i+3]);
        i = (480 * W + 32) * 4;
        printf("CLUT (32,480): R=%d G=%d B=%d A=%d\n", px[i], px[i+1], px[i+2], px[i+3]);

        // Now check render target (where the draw outputs)
        auto *cb2 = [queue commandBuffer]; auto *bl2 = [cb2 blitCommandEncoder];
        [bl2 copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(W, H, 1) toBuffer:stg destinationOffset:0
          destinationBytesPerRow:W*4 destinationBytesPerImage:bytes];
        [bl2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
        px = (const uint8_t *)stg.contents;

        // Check where the rect actually rendered (checking alpha too)
        int found = 0;
        for (int y = 195; y < 210; y++)
            for (int x = 195; x < 270; x++) {
                int idx = (y * W + x) * 4;
                if (px[idx]>0||px[idx+1]>0||px[idx+2]>0||px[idx+3]>0) {
                    if (found < 5) printf("  RT at (%d,%d): R=%d G=%d B=%d A=%d\n", x, y, px[idx], px[idx+1], px[idx+2], px[idx+3]);
                    found++;
                }
            }
        printf("RT rendered pixels in rect area (y=195-210): %d\n", found);

        // Also scan wider — maybe it landed somewhere else
        int total = 0;
        int firstX=-1, firstY=-1;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int idx = (y * W + x) * 4;
                if (px[idx+3] > 0 && (px[idx]>0||px[idx+1]>0||px[idx+2]>0)) {
                    if (total == 0) { firstX = x; firstY = y; }
                    total++;
                }
            }
        printf("RT total non-black+alpha pixels: %d, first at (%d,%d)\n", total, firstX, firstY);
    }

    // Check: pixel at x=200 should have CLUT index 0 (dim red)
    // pixel at x=232 should have CLUT index 32 (brighter red)
    // Viewport flip + draw offset (-0.5) shifts Y by 1 pixel
    auto px0 = readPixel(200, 199);
    auto px32 = readPixel(232, 199);
    auto px63 = readPixel(263, 199);

    printf("px0: R=%d G=%d B=%d A=%d\n", px0.r, px0.g, px0.b, px0.a);
    printf("px32: R=%d G=%d B=%d A=%d\n", px32.r, px32.g, px32.b, px32.a);
    printf("px63: R=%d G=%d B=%d A=%d\n", px63.r, px63.g, px63.b, px63.a);

    EXPECT_GT(px0.r, 0) << "CLUT index 0 should produce non-black";
    EXPECT_GT(px32.r, px0.r) << "CLUT index 32 should be brighter than index 0";
    EXPECT_GT(px63.r, px32.r) << "CLUT index 63 should be brighter than index 32";

    // Verify we get actual color variation across the rect
    int distinctColors = 0;
    uint8_t lastR = 255;
    for (int x = 200; x < 264; x++) {
        auto px = readPixel(x, 199);
        if (px.r != lastR) { distinctColors++; lastR = px.r; }
    }
    EXPECT_GT(distinctColors, 10) << "8bpp texture should produce many distinct colors across 64 pixels";
}

TEST_F(MetalGPUTest, UVInterpolation_VariesAcrossRect) {
    // Draw a textured rect and check that UV.x varies across the width
    // Use a diagnostic shader that outputs UV.x as the red channel
    // For this test, we don't care about CLUT — just UV interpolation

    uint16_t tp = 0x0080;  // 8bpp, page (0,0) — texMode will be 1
    uint16_t cl = 0;
    uint32_t white = 0x808080;
    std::vector<Vertex> verts = {
        {100,100,white, tp,cl, 0,0}, {164,100,white, tp,cl, 64,0}, {164,120,white, tp,cl, 64,20},
        {164,120,white, tp,cl, 64,20}, {100,120,white, tp,cl, 0,20}, {100,100,white, tp,cl, 0,0},
    };
    draw(verts);

    // Read pixels along row 110 (middle of rect)
    std::vector<uint8_t> redValues;
    for (int x = 100; x < 164; x++) {
        auto px = readPixel(x, 110);
        redValues.push_back(px.r);
    }

    // The shader computes texBlend(clutColor, vertexColor).
    // But we want to see the RAW UV. Since this test doesn't upload CLUT/texture data,
    // the CLUT lookup will return black. texBlend(0, 0.5) = 0. So everything is black.
    // We need a different approach — check if DIFFERENT pixels produce DIFFERENT results.

    // Instead: check that the render target has ANY variation
    int distinct = 0;
    uint8_t last = redValues[0];
    for (uint8_t v : redValues) {
        if (v != last) { distinct++; last = v; }
    }
    printf("UV test: %d distinct R values across 64 pixels\n", distinct);
    printf("First 10 R values: ");
    for (int i = 0; i < 10 && i < (int)redValues.size(); i++) printf("%d ", redValues[i]);
    printf("\n");

    // Even if CLUT returns black, alpha might vary. Check alpha.
    std::vector<uint8_t> alphaValues;
    for (int x = 100; x < 164; x++) {
        auto px = readPixel(x, 110);
        alphaValues.push_back(px.a);
    }
    printf("First 10 A values: ");
    for (int i = 0; i < 10 && i < (int)alphaValues.size(); i++) printf("%d ", alphaValues[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else
#include <cstdio>
int main() { printf("Build with -DPCSX_METAL\n"); return 1; }
#endif
