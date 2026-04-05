// Standalone Metal GPU 4bpp CLUT test
#ifdef PCSX_METAL
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>

static const char *shaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct VIn { int2 pos [[attribute(0)]]; uint color [[attribute(1)]]; ushort tp [[attribute(2)]]; ushort clut [[attribute(3)]]; ushort2 uv [[attribute(4)]]; };
struct VOut { float4 position [[position]]; float2 tc; int2 clutBase [[flat]]; int2 tpBase [[flat]]; int texMode [[flat]]; };
struct Uniforms { float2 drawOffset; int4 texWindow; float4 blend; float4 blendOpaque; };

vertex VOut vmain(VIn in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    VOut o;
    o.position = float4((float(in.pos.x)+u.drawOffset.x)/512.0-1.0, 1.0-(float(in.pos.y)+u.drawOffset.y)/256.0, 0, 1);
    o.texMode = (in.tp >> 7) & 3;
    o.tc = float2(in.uv);
    o.tpBase = int2((in.tp & 0xf)*64, ((in.tp>>4)&1)*256);
    o.clutBase = int2((in.clut & 0x3f)*16, in.clut >> 6);
    return o;
}

// Diagnostic fragment: output intermediate values as pixel colors
struct FOut { float4 color [[color(0)]]; };

fragment FOut fmain(VOut in [[stage_in]],
                    texture2d<float, access::read> vram [[texture(0)]],
                    constant Uniforms &u [[buffer(1)]]) {
    FOut out;
    int2 UV = int2(floor(in.tc + float2(0.0001))) & int2(0xff);

    // Read the texel that contains the 4bpp indices
    int2 tc = int2(UV.x >> 2, UV.y) + in.tpBase;
    tc &= int2(1023, 511);
    float4 texel = vram.read(uint2(tc));

    // Reconstruct the 16-bit value (same as sample16)
    int r = int(floor(texel.r * 31.0 + 0.5));
    int g = int(floor(texel.g * 31.0 + 0.5));
    int b = int(floor(texel.b * 31.0 + 0.5));
    int msb = int(ceil(texel.a)) << 15;
    int word16 = r | (g << 5) | (b << 10) | msb;

    // Extract the 4bpp index
    int shift = (UV.x & 3) << 2;
    int clutIndex = (word16 >> shift) & 0xf;

    // Read CLUT entry
    int2 clutCoord = int2(in.clutBase.x + clutIndex, in.clutBase.y);
    clutCoord &= int2(1023, 511);
    float4 clutColor = vram.read(uint2(clutCoord));

    // Output diagnostics encoded as color:
    // R = texel.r raw (0-255 range), G = clutIndex / 15, B = clutColor.r
    // This lets us see what the GPU computes at each stage.
    // For the first pixel (UV.x=0): shift=0, index should be (word16 & 0xf)
    out.color = float4(texel.r, float(clutIndex) / 15.0, clutColor.r, 1.0);
    return out;
}
)";

struct Vertex { int32_t x, y; uint32_t color; uint16_t texpage, clut; uint16_t u, v; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:shaderSrc] options:nil error:&err];
        if (!lib) { fprintf(stderr, "Shader error: %s\n", err.localizedDescription.UTF8String); return 1; }

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

        auto *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1024 height:512 mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> vramRT = [dev newTextureWithDescriptor:td];
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> vramSample = [dev newTextureWithDescriptor:td];

        // Upload CLUT at (0, 480): 16 entries with increasing red
        uint32_t clutRGBA[16];
        for (int i = 0; i < 16; i++) {
            uint8_t r = i * 255 / 31;  // NOTE: using i not i*2
            clutRGBA[i] = r | (0 << 8) | (0 << 16) | (0xff << 24);
        }
        // Make entry 0 NOT black (so it doesn't get discarded)
        clutRGBA[0] = 50 | (50 << 8) | (50 << 16) | (0xff << 24);
        [vramRT replaceRegion:MTLRegionMake2D(0, 480, 16, 1) mipmapLevel:0 withBytes:clutRGBA bytesPerRow:16*4];

        // Upload 4bpp texture at (0,0): one 16-bit word containing indices 5,6,7,8
        // word = 5 | (6<<4) | (7<<8) | (8<<12) = 0x8765
        uint16_t packed = 0x8765;
        uint8_t r8 = (packed & 0x1f) * 255 / 31;
        uint8_t g8 = ((packed >> 5) & 0x1f) * 255 / 31;
        uint8_t b8 = ((packed >> 10) & 0x1f) * 255 / 31;
        uint8_t a8 = (packed & 0x8000) ? 0xff : 0x00;
        uint32_t texRGBA = r8 | (g8 << 8) | (b8 << 16) | (a8 << 24);
        [vramRT replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&texRGBA bytesPerRow:4];

        printf("Packed word: 0x%04x → RGBA8(%d,%d,%d,%d)\n", packed, r8, g8, b8, a8);
        printf("Expected indices: px0=%d px1=%d px2=%d px3=%d\n", packed&0xf, (packed>>4)&0xf, (packed>>8)&0xf, (packed>>12)&0xf);

        // Sync
        auto *cb1 = [queue commandBuffer];
        auto *bl1 = [cb1 blitCommandEncoder];
        [bl1 copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1024,512,1)
                   toTexture:vramSample destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [bl1 endEncoding]; [cb1 commit]; [cb1 waitUntilCompleted];

        // Draw a 4-pixel wide rect: PSX (200,200)-(204,201), UV (0,0)-(4,1)
        // texpage = 0x0000 (4bpp, page 0,0), clut at (0,480)
        uint16_t tp = 0x0000;
        uint16_t cl = (480 << 6); // clutY=480, clutX=0
        Vertex verts[6] = {
            {200,200, 0x808080, tp, cl, 0, 0},
            {204,200, 0x808080, tp, cl, 4, 0},
            {204,201, 0x808080, tp, cl, 4, 1},
            {204,201, 0x808080, tp, cl, 4, 1},
            {200,201, 0x808080, tp, cl, 0, 1},
            {200,200, 0x808080, tp, cl, 0, 0},
        };

        struct { float dox,doy; int twmx,twmy,twox,twoy; float br,bg,bb,ba,bor,bog,bob,boa; } u = {0.5f,-0.5f, ~0,~0, 0,0, 1,1,1,0, 1,1,1,0};

        auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = vramRT;
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        auto *cb2 = [queue commandBuffer];
        auto *enc = [cb2 renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:pso];
        [enc setVertexBytes:verts length:sizeof(verts) atIndex:0];
        [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
        [enc setFragmentTexture:vramSample atIndex:0];
        MTLScissorRect sc = {0,0,1024,512};
        [enc setScissorRect:sc];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [enc endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];

        // Readback
        constexpr size_t vramBytes = 1024*512*4;
        auto *stg = [dev newBufferWithLength:vramBytes options:MTLResourceStorageModeShared];
        auto *cb3 = [queue commandBuffer];
        auto *bl3 = [cb3 blitCommandEncoder];
        [bl3 copyFromTexture:vramRT sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1024,512,1) toBuffer:stg destinationOffset:0 destinationBytesPerRow:1024*4 destinationBytesPerImage:vramBytes];
        [bl3 endEncoding]; [cb3 commit]; [cb3 waitUntilCompleted];
        auto *px = (const uint8_t *)stg.contents;

        // The rect should be around row 200 (with Y flip)
        printf("\nOutput pixels around (200,200):\n");
        for (int y = 198; y < 203; y++) {
            for (int x = 198; x < 206; x++) {
                int i = (y*1024+x)*4;
                if (px[i]>0 || px[i+1]>0 || px[i+2]>0) {
                    // Decode diagnostic: R=texel.r(0-255), G=clutIndex/15(0-255), B=clutColor.r(0-255)
                    float texelR = px[i] / 255.0f;
                    float clutIdxNorm = px[i+1] / 255.0f;
                    int clutIdx = (int)(clutIdxNorm * 15.0f + 0.5f);
                    float clutR = px[i+2] / 255.0f;
                    printf("  (%d,%d): raw=(%d,%d,%d,%d) → texelR=%.3f clutIdx=%d clutColorR=%.3f\n",
                           x, y, px[i], px[i+1], px[i+2], px[i+3], texelR, clutIdx, clutR);
                }
            }
        }

        printf("DONE\n");
    }
    return 0;
}
#else
int main() { return 1; }
#endif
