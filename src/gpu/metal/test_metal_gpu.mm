// Test: replicate the exact emulator flow for textured rendering.
// Upload data via the same blit path as partialUpdateVRAM, sync, draw, readback.
#ifdef PCSX_METAL
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static const int W = 1024, H = 512;

// Same shader as gpu_metal.mm (simplified — 8bpp only for test)
static NSString *const shaderSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VIn { int2 pos [[attribute(0)]]; uint color [[attribute(1)]]; ushort tp [[attribute(2)]]; ushort clut [[attribute(3)]]; ushort2 uv [[attribute(4)]]; };
struct VOut { float4 position [[position]]; float4 vcolor; float2 tc; int2 clutBase [[flat]]; int2 tpBase [[flat]]; int texMode [[flat]]; };
struct Uniforms { float2 drawOffset; int4 texWindow; float4 blend; float4 blendOpaque; };

vertex VOut vmain(VIn in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    VOut o;
    o.position = float4((float(in.pos.x)+u.drawOffset.x)/512.0-1.0, 1.0-(float(in.pos.y)+u.drawOffset.y)/256.0, 0, 1);
    o.vcolor = float4(float(in.color&0xffu)/255.0, float((in.color>>8u)&0xffu)/255.0, float((in.color>>16u)&0xffu)/255.0, 1);
    o.texMode = (in.tp >> 7) & 3;
    o.tc = float2(in.uv);
    o.tpBase = int2((in.tp & 0xf)*64, ((in.tp>>4)&1)*256);
    o.clutBase = int2((in.clut & 0x3f)*16, in.clut >> 6);
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
    int2 UV = int2(floor(in.tc + float2(0.0001))) & int2(0xff);
    UV = (UV & u.texWindow.xy) | u.texWindow.zw;
    // 8bpp path — diagnostic: output index and UV as colors
    int2 tc = int2(UV.x >> 1, UV.y) + in.tpBase;
    int s = sample16(vram, tc);
    int shift = (UV.x & 1) << 3;
    int idx = (s >> shift) & 0xff;

    // Encode diagnostics: R = idx/255, G = UV.x/64, B = sample16 low byte / 255
    out.color = float4(float(idx) / 255.0, float(UV.x) / 64.0, float(s & 0xff) / 255.0, 1.0);
    return out;
}
)MSL";

struct Vertex { int32_t x, y; uint32_t color; uint16_t texpage, clut; uint16_t u, v; };

// Replicate partialUpdateVRAM: convert 16-bit to RGBA8, blit to render target
void uploadToRT(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLTexture> rt,
                int x, int y, int w, int h, const uint16_t *pixels) {
    std::vector<uint32_t> rgba(w * h);
    for (int i = 0; i < w * h; i++) {
        uint16_t px = pixels[i];
        uint8_t r = ((px >> 0) & 0x1f) * 255 / 31;
        uint8_t g = ((px >> 5) & 0x1f) * 255 / 31;
        uint8_t b = ((px >> 10) & 0x1f) * 255 / 31;
        uint8_t a = (px & 0x8000) ? 0xff : 0x00;
        rgba[i] = r | (g << 8) | (b << 16) | (a << 24);
    }
    size_t byteSize = rgba.size() * 4;
    auto *staging = [dev newBufferWithBytes:rgba.data() length:byteSize options:MTLResourceStorageModeShared];
    auto *cb = [queue commandBuffer];
    auto *bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:w*4 sourceBytesPerImage:byteSize
            sourceSize:MTLSizeMake(w, h, 1)
             toTexture:rt destinationSlice:0 destinationLevel:0
      destinationOrigin:MTLOriginMake(x, y, 0)];
    [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:shaderSrc options:nil error:&err];
        if (!lib) { fprintf(stderr, "Shader: %s\n", err.localizedDescription.UTF8String); return 1; }

        auto *vd = [[MTLVertexDescriptor alloc] init];
        vd.attributes[0].format = MTLVertexFormatInt2;    vd.attributes[0].offset = 0;  vd.attributes[0].bufferIndex = 0;
        vd.attributes[1].format = MTLVertexFormatUInt;    vd.attributes[1].offset = 8;  vd.attributes[1].bufferIndex = 0;
        vd.attributes[2].format = MTLVertexFormatUShort;  vd.attributes[2].offset = 12; vd.attributes[2].bufferIndex = 0;
        vd.attributes[3].format = MTLVertexFormatUShort;  vd.attributes[3].offset = 14; vd.attributes[3].bufferIndex = 0;
        vd.attributes[4].format = MTLVertexFormatUShort2; vd.attributes[4].offset = 16; vd.attributes[4].bufferIndex = 0;
        vd.layouts[0].stride = sizeof(Vertex);
        auto *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = [lib newFunctionWithName:@"vmain"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fmain"];
        pd.vertexDescriptor = vd;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> pso = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso) { fprintf(stderr, "PSO: %s\n", err.localizedDescription.UTF8String); return 1; }

        // Create textures — same as emulator
        auto *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> vramRT = [dev newTextureWithDescriptor:td];
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> vramSample = [dev newTextureWithDescriptor:td];

        // ---- Setup: Upload 8bpp texture data at texpage (11*64, 0) = (704, 0) ----
        // 8bpp: each 16-bit VRAM word holds 2 pixel indices
        // Create 64x64 texture with indices 0-255 repeating
        int texW = 32; // 32 VRAM words = 64 8bpp pixels
        int texH = 64;
        std::vector<uint16_t> texData(texW * texH);
        for (int row = 0; row < texH; row++) {
            for (int col = 0; col < texW; col++) {
                uint8_t idx0 = (row * texW + col) * 2;
                uint8_t idx1 = idx0 + 1;
                // Pack two 8-bit indices into a 16-bit word as raw bytes:
                // low byte = idx0, high byte = idx1
                // As 5:5:5:1: we need to store raw 16-bit, so encode idx0|(idx1<<8) as 16-bit
                uint16_t packed = idx0 | (idx1 << 8);
                texData[row * texW + col] = packed;
            }
        }
        printf("Uploading %dx%d 8bpp texture at (%d, %d)\n", texW, texH, 704, 0);
        uploadToRT(dev, queue, vramRT, 704, 0, texW, texH, texData.data());

        // ---- Upload CLUT at (512, 241) — 256 entries ----
        std::vector<uint16_t> clutData(256);
        for (int i = 0; i < 256; i++) {
            // Rainbow gradient: R varies with i, G fixed, B inverse
            int r = (i * 31 / 255) & 0x1f;
            int g = 15;
            int b = ((255 - i) * 31 / 255) & 0x1f;
            clutData[i] = r | (g << 5) | (b << 10) | 0x8000;
        }
        // Make entry 0 non-black so it's visible
        clutData[0] = 1 | (1 << 5) | (1 << 10) | 0x8000;
        printf("Uploading 256-entry CLUT at (512, 241)\n");
        uploadToRT(dev, queue, vramRT, 512, 241, 256, 1, clutData.data());

        // Verify render target has data
        constexpr size_t vramBytes = W * H * 4;
        auto *stg = [dev newBufferWithLength:vramBytes options:MTLResourceStorageModeShared];
        {
            auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
            [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(W,H,1)
                      toBuffer:stg destinationOffset:0 destinationBytesPerRow:W*4 destinationBytesPerImage:vramBytes];
            [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        auto *px = (const uint8_t *)stg.contents;
        printf("RT at (704,0): R=%d G=%d B=%d A=%d\n", px[(0*W+704)*4], px[(0*W+704)*4+1], px[(0*W+704)*4+2], px[(0*W+704)*4+3]);
        printf("RT at (512,241): R=%d G=%d B=%d A=%d\n", px[(241*W+512)*4], px[(241*W+512)*4+1], px[(241*W+512)*4+2], px[(241*W+512)*4+3]);

        // ---- Sync: blit RT → sample (same as syncVRAMTexture) ----
        {
            auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
            [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(W,H,1)
                      toTexture:vramSample destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
            [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        // Verify sample texture
        {
            auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
            [bl copyFromTexture:vramSample sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(W,H,1)
                      toBuffer:stg destinationOffset:0 destinationBytesPerRow:W*4 destinationBytesPerImage:vramBytes];
            [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        px = (const uint8_t *)stg.contents;
        printf("Sample at (704,0): R=%d G=%d B=%d A=%d\n", px[(0*W+704)*4], px[(0*W+704)*4+1], px[(0*W+704)*4+2], px[(0*W+704)*4+3]);
        printf("Sample at (512,241): R=%d G=%d B=%d A=%d\n", px[(241*W+512)*4], px[(241*W+512)*4+1], px[(241*W+512)*4+2], px[(241*W+512)*4+3]);

        // ---- Draw: 64x64 textured rect at (200,200)-(264,264) ----
        // texpage: page X=11 (704/64), Y=0, depth=1 (8bpp) → tp = 11 | (0<<4) | (1<<7) = 0x008b
        // clut at (512, 241): clutraw = (512/16 & 0x3f) | (241 << 6) = 32 | (241<<6) = 32 | 15424 = 15456 = 0x3c60... wait
        // clutX = (cl & 0x3f) * 16 → need cl & 0x3f = 512/16 = 32 → lower 6 bits = 32
        // clutY = cl >> 6 → need 241 → cl = 32 | (241 << 6) = 32 | 15424 = 15456
        uint16_t tp = 0x008b;  // page X=11, Y=0, depth=8bpp
        uint16_t cl = 32 | (241 << 6);  // CLUT at (512, 241)
        uint32_t white = 0x808080;
        Vertex verts[6] = {
            {200,200,white, tp,cl, 0,0},
            {264,200,white, tp,cl, 64,0},
            {264,264,white, tp,cl, 64,64},
            {264,264,white, tp,cl, 64,64},
            {200,264,white, tp,cl, 0,64},
            {200,200,white, tp,cl, 0,0},
        };

        struct { float dox,doy; int twmx,twmy,twox,twoy; float br,bg,bb,ba,bor,bog,bob,boa; } u = {0.5f,-0.5f, ~0,~0, 0,0, 1,1,1,0, 1,1,1,0};

        auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = vramRT;
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        auto *cb = [queue commandBuffer];
        auto *enc = [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        // NO viewport flip — Y negated in vertex shader instead
        auto *vertBuf = [dev newBufferWithBytes:verts length:sizeof(verts) options:MTLResourceStorageModeShared];
        [enc setVertexBuffer:vertBuf offset:0 atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentTexture:vramSample atIndex:0];
        MTLScissorRect sc = {0,0,(NSUInteger)W,(NSUInteger)H};
        [enc setScissorRect:sc];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

        // ---- Readback and check ----
        {
            auto *cb = [queue commandBuffer]; auto *bl = [cb blitCommandEncoder];
            [bl copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(W,H,1)
                      toBuffer:stg destinationOffset:0 destinationBytesPerRow:W*4 destinationBytesPerImage:vramBytes];
            [bl endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
        px = (const uint8_t *)stg.contents;

        // Scan for rendered pixels in the rect area
        int found = 0;
        for (int y = 195; y < 270; y++) {
            for (int x = 195; x < 270; x++) {
                int i = (y*W+x)*4;
                if (px[i]>0 || px[i+1]>0 || px[i+2]>0) {
                    if (found < 8)
                        printf("  Textured pixel at (%d,%d): R=%d G=%d B=%d A=%d\n", x, y, px[i], px[i+1], px[i+2], px[i+3]);
                    found++;
                }
            }
        }
        printf("Found %d non-black pixels in rect area (195-270)\n", found);

        // Check a few specific spots
        int total = 0;
        for (int i = 0; i < W*H; i++) {
            if (px[i*4]>0||px[i*4+1]>0||px[i*4+2]>0) total++;
        }
        printf("Total non-black in VRAM: %d\n", total);

        printf("TEST DONE\n");
    }
    return 0;
}
#else
int main() { return 1; }
#endif
