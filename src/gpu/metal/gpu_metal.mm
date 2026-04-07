/***************************************************************************
 *   Copyright (C) 2026 PCSX-Redux authors                                 *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.           *
 ***************************************************************************/

#ifdef PCSX_METAL

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "gpu/metal/gpu_metal.h"
#include "core/psxemulator.h"
#include "core/system.h"
#include "core/ui.h"

// ---------------------------------------------------------------------------
// Metal Shading Language — ported from the OpenGL GPU's GLSL 330 shaders
// ---------------------------------------------------------------------------
static NSString *const s_metalShaderSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    int2   position [[attribute(0)]];
    uint   color    [[attribute(1)]];
    ushort texpage  [[attribute(2)]];
    ushort clut     [[attribute(3)]];
    ushort2 uv      [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 vertexColor;
    float2 texCoords;
    int2   clutBase   [[flat]];
    int2   texpageBase [[flat]];
    int    texMode    [[flat]];
};

struct Uniforms {
    float drawOffsetX;
    float drawOffsetY;
    int texWindowMaskX;
    int texWindowMaskY;
    int texWindowOffX;
    int texWindowOffY;
    float blendR, blendG, blendB, blendA;
    float blendOpaqueR, blendOpaqueG, blendOpaqueB, blendOpaqueA;
    int setMask;    // force MSB=1 on written pixels
    int checkMask;  // skip writes to pixels with MSB=1
};

vertex VertexOut vertexMain(VertexIn in [[stage_in]],
                            constant Uniforms &u [[buffer(1)]]) {
    VertexOut out;

    float x = float(in.position.x);
    float y = float(in.position.y);
    float xx = (x + u.drawOffsetX) / 512.0 - 1.0;
    // Metal NDC Y+ is down (top-left origin). Divide by -256 so VRAM Y=0
    // maps to clip +1 (top) and Y=511 maps to clip -1 (bottom).
    float yy = (y + u.drawOffsetY) / -256.0 + 1.0;

    float red   = float(in.color & 0xffu);
    float green = float((in.color >> 8u) & 0xffu);
    float blue  = float((in.color >> 16u) & 0xffu);

    out.position    = float4(xx, yy, 0.0, 1.0);
    out.vertexColor = float4(red / 255.0, green / 255.0, blue / 255.0, 1.0);

    if ((in.texpage & 0x8000) != 0) {
        out.texMode = 4;  // untextured
    } else {
        out.texMode     = (in.texpage >> 7) & 3;
        out.texCoords   = float2(in.uv);
        out.texpageBase = int2((in.texpage & 0xf) * 64, ((in.texpage >> 4) & 0x1) * 256);
        out.clutBase    = int2((in.clut & 0x3f) * 16, in.clut >> 6);
    }
    return out;
}

// --- helpers ----------------------------------------------------------------

// Read raw 16-bit value from R16Uint VRAM texture
static uint readVRAM16(texture2d<uint, access::read> tex, int2 c) {
    c &= int2(1023, 511);
    return tex.read(uint2(c)).r;
}

// Convert raw 16-bit PSX color (5:5:5:1) to float4 RGBA
static float4 color16toFloat4(uint raw) {
    float r = float(raw & 0x1fu) / 31.0;
    float g = float((raw >> 5u) & 0x1fu) / 31.0;
    float b = float((raw >> 10u) & 0x1fu) / 31.0;
    float a = float((raw >> 15u) & 1u);
    return float4(r, g, b, a);
}

static float4 texBlend(float4 c1, float4 c2) {
    float4 ret = (c1 * c2) / (128.0 / 255.0);
    ret.a = 1.0;
    return ret;
}

// --- fragment ---------------------------------------------------------------

struct FragmentOut {
    float4 color      [[color(0), index(0)]];
    float4 blendColor [[color(0), index(1)]];
};

fragment FragmentOut fragmentMain(VertexOut in [[stage_in]],
                                  texture2d<uint, access::read> vramTex [[texture(0)]],
                                  constant Uniforms &u [[buffer(1)]]) {
    FragmentOut out;

    // TODO: checkMask needs a dedicated stencil buffer — using RT alpha doesn't work
    // because VRAM texture regions have MSB=1 from CPU uploads, causing false positives.
    // For now, checkMask is tracked in uniforms but not enforced in the shader.

    // Untextured primitive
    if (in.texMode == 4) {
        float a = u.setMask ? 1.0 : 0.0;
        out.color = float4(in.vertexColor.rgb * u.blendR, a);
        out.blendColor = float4(0.0, 0.0, 0.0, u.blendA);
        return out;
    }

    int2 UV = int2(floor(in.texCoords + float2(0.0001))) & int2(0xff);
    UV = (UV & int2(u.texWindowMaskX, u.texWindowMaskY)) | int2(u.texWindowOffX, u.texWindowOffY);

    // Sample texture and look up CLUT
    float4 texColor;
    if (in.texMode == 0) {  // 4bpp
        int2 tc = int2(UV.x >> 2, UV.y) + in.texpageBase;
        uint word = readVRAM16(vramTex, tc);
        int idx = (word >> ((UV.x & 3) << 2)) & 0xf;
        texColor = color16toFloat4(readVRAM16(vramTex, int2(in.clutBase.x + idx, in.clutBase.y)));
    } else if (in.texMode == 1) {  // 8bpp
        int2 tc = int2(UV.x >> 1, UV.y) + in.texpageBase;
        uint word = readVRAM16(vramTex, tc);
        int idx = (word >> ((UV.x & 1) << 3)) & 0xff;
        texColor = color16toFloat4(readVRAM16(vramTex, int2(in.clutBase.x + idx, in.clutBase.y)));
    } else {  // 16bpp
        texColor = color16toFloat4(readVRAM16(vramTex, UV + in.texpageBase));
    }

    // Raw 16-bit value 0x0000 = fully transparent (never drawn).
    // 0x8000 (black with MSB=1) IS drawn — it produces a dimming effect
    // when semi-transparent blending is active.
    if (all(texColor.rgb == float3(0.0)) && texColor.a < 0.5) discard_fragment();

    // Apply texture blending (modulate texture by vertex color)
    float4 color = texBlend(texColor, in.vertexColor);

    // Per-pixel semi-transparency: MSB (alpha) of the texel determines
    // whether this pixel is semi-transparent or opaque.
    float maskAlpha = u.setMask ? 1.0 : texColor.a;
    bool semiTransparent = texColor.a >= 0.5;
    if (semiTransparent) {
        out.color = float4(color.rgb * u.blendR, maskAlpha);
        out.blendColor = float4(0.0, 0.0, 0.0, u.blendA);
    } else {
        float a = u.setMask ? 1.0 : 0.0;
        out.color = float4(color.rgb, a);
        out.blendColor = float4(0.0, 0.0, 0.0, 0.0);
    }
    return out;
}
)MSL";

// ---------------------------------------------------------------------------
// Compute shader: convert RGBA8 render target → R16Uint sample texture (on GPU)
// ---------------------------------------------------------------------------
static NSString *const s_syncComputeSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rgba8ToR16(texture2d<float, access::read>  src [[texture(0)]],
                       texture2d<uint, access::write>   dst [[texture(1)]],
                       constant uint2 &offset [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint2 pos = gid + offset;
    if (pos.x >= 1024 || pos.y >= 512) return;
    float4 c = src.read(pos);
    uint r = uint(c.r * 31.0 + 0.5);
    uint g = uint(c.g * 31.0 + 0.5);
    uint b = uint(c.b * 31.0 + 0.5);
    uint a = uint(ceil(c.a));
    dst.write(uint4(r | (g << 5) | (b << 10) | (a << 15), 0, 0, 0), pos);
}
)MSL";

// ---------------------------------------------------------------------------
// Metal objects stored as ObjC types (hidden from the header)
// ---------------------------------------------------------------------------
struct PCSX::MetalGPU::MetalObjects {
    id<MTLDevice>               device = nil;
    id<MTLCommandQueue>         commandQueue = nil;
    id<MTLTexture>              vramRenderTarget = nil;   // render target (1024x512 RGBA8)
    id<MTLTexture>              vramSampleTexture = nil;  // R16Uint — raw 16-bit VRAM for shader reads
    id<MTLTexture>              displayTexture = nil;     // decoded display output (for 24-bit mode)
    id<MTLTexture>              displaySnapshot = nil;    // snapshot of VRAM for ImGui to sample
    id<MTLBuffer>               vertexBuffer = nil;       // streaming vertex data
    id<MTLRenderPipelineState>  psoOpaque = nil;
    id<MTLRenderPipelineState>  psoBlendAdd = nil;
    id<MTLRenderPipelineState>  psoBlendRevSub = nil;
    id<MTLComputePipelineState> syncComputePSO = nil;     // RGBA8 → R16Uint conversion
    MTLScissorRect              scissor = {0, 0, 1024, 512};
    bool                        display24bit = false;

    // Per-frame command buffer — created at frame start, committed at vblank.
    // All draws, blits, and syncs within a frame share this command buffer.
    id<MTLCommandBuffer>        frameCmdBuf = nil;
    bool                        renderEncoderActive = false;
    id<MTLRenderCommandEncoder> renderEncoder = nil;

    void ensureNoRenderEncoder() {
        if (renderEncoderActive) {
            [renderEncoder endEncoding];
            renderEncoder = nil;
            renderEncoderActive = false;
        }
    }

    void ensureFrameCmdBuf() {
        if (!frameCmdBuf) {
            frameCmdBuf = [commandQueue commandBuffer];
        }
    }

    void commitAndWait() {
        ensureNoRenderEncoder();
        if (frameCmdBuf) {
            [frameCmdBuf commit];
            [frameCmdBuf waitUntilCompleted];
            frameCmdBuf = nil;
        }
    }
};

// ---------------------------------------------------------------------------
// Helper to build a PSO with a specific blend configuration
// ---------------------------------------------------------------------------
static id<MTLRenderPipelineState> makePSO(id<MTLDevice> device,
                                          id<MTLFunction> vertFunc,
                                          id<MTLFunction> fragFunc,
                                          MTLVertexDescriptor *vertDesc,
                                          bool blendEnabled,
                                          MTLBlendOperation blendOp) {
    auto *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction   = vertFunc;
    desc.fragmentFunction = fragFunc;
    desc.vertexDescriptor = vertDesc;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

    if (blendEnabled) {
        desc.colorAttachments[0].blendingEnabled          = YES;
        desc.colorAttachments[0].rgbBlendOperation        = blendOp;
        desc.colorAttachments[0].alphaBlendOperation      = MTLBlendOperationAdd;
        // Source color is premultiplied by src_alpha_factor in the shader
        desc.colorAttachments[0].sourceRGBBlendFactor      = MTLBlendFactorOne;
        // Destination weight comes from second output's alpha channel
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorSource1Alpha;
        desc.colorAttachments[0].sourceAlphaBlendFactor    = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
    }

    NSError *err = nil;
    auto pso = [device newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!pso) {
        NSLog(@"Metal PSO creation failed: %@", err);
    }
    return pso;
}

// ---------------------------------------------------------------------------
// Per-frame GPU command log
// ---------------------------------------------------------------------------
static struct {
    int polysUntex = 0, polysTex = 0;
    int lines = 0;
    int rectsUntex = 0, rectsTex = 0;
    int fastFills = 0;
    int blitVV = 0;
    int cpuUploads = 0;
    int totalVerts = 0;
    int batches = 0;
    int batchesOpaque = 0;
    int batchesSemiTrans = 0;
    int frameNum = 0;
    int scissorX = 0, scissorY = 0, scissorW = 0, scissorH = 0;
    bool enabled = false;  // set to true to enable periodic GPU command logging
    int logInterval = 120;

    void reset() {
        polysUntex = polysTex = lines = rectsUntex = rectsTex = 0;
        fastFills = blitVV = cpuUploads = totalVerts = batches = 0;
        batchesOpaque = batchesSemiTrans = 0;
    }
    void dump(PCSX::GPU::Display &disp) {
        frameNum++;
        if (!enabled || (frameNum % logInterval != 0)) { reset(); return; }
        fprintf(stderr, "[GPU f%d] en=%d d=%d start=(%d,%d) sz=(%d,%d) | "
                "poly:%d/%d line:%d rect:%d/%d fill:%d blit:%d upload:%d | "
                "verts=%d batches=%d (opaque:%d semi:%d) | scissor=(%d,%d,%d,%d)\n",
                frameNum, disp.enabled, disp.info.depth,
                disp.start.x(), disp.start.y(), disp.size.x(), disp.size.y(),
                polysUntex, polysTex, lines, rectsUntex, rectsTex,
                fastFills, blitVV, cpuUploads,
                totalVerts, batches, batchesOpaque, batchesSemiTrans,
                (int)scissorX, (int)scissorY, (int)scissorW, (int)scissorH);
        reset();
    }
} s_log;

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<PCSX::GPU> PCSX::GPU::getMetal() { return std::make_unique<PCSX::MetalGPU>(); }

PCSX::MetalGPU::MetalGPU() { m_vertices.resize(c_vertexBufferSize); }
PCSX::MetalGPU::~MetalGPU() { shutdown(); }

// ---------------------------------------------------------------------------
// Init / Shutdown
// ---------------------------------------------------------------------------
int PCSX::MetalGPU::initBackend(UI *) {
    m_mtl = new MetalObjects();
    m_mtl->device = MTLCreateSystemDefaultDevice();
    if (!m_mtl->device) { g_system->message("Metal GPU: no device\n"); return -1; }
    m_mtl->commandQueue = [m_mtl->device newCommandQueue];

    // --- VRAM textures (render target + sample copy) ---
    auto *texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                      width:c_vramWidth
                                                                     height:c_vramHeight
                                                                  mipmapped:NO];
    texDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModeShared;
    m_mtl->vramRenderTarget = [m_mtl->device newTextureWithDescriptor:texDesc];

    // Sample texture: R16Uint stores raw 16-bit PSX pixels.
    // No float conversion — the shader reads uint and extracts bytes/nibbles directly.
    auto *sampleDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Uint
                                                                         width:c_vramWidth
                                                                        height:c_vramHeight
                                                                     mipmapped:NO];
    sampleDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    sampleDesc.storageMode = MTLStorageModeShared;
    m_mtl->vramSampleTexture = [m_mtl->device newTextureWithDescriptor:sampleDesc];

    // Display texture for 24-bit decode (shared storage so CPU can write)
    auto *displayDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                          width:c_vramWidth
                                                                         height:c_vramHeight
                                                                      mipmapped:NO];
    displayDesc.usage = MTLTextureUsageShaderRead;
    displayDesc.storageMode = MTLStorageModeShared;
    m_mtl->displayTexture = [m_mtl->device newTextureWithDescriptor:displayDesc];

    // Snapshot texture — a stable copy of the display region for ImGui to sample.
    // Managed storage lets the GPU blit into it and ImGui sample from it without races.
    auto *snapDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                       width:c_vramWidth
                                                                      height:c_vramHeight
                                                                   mipmapped:NO];
    snapDesc.usage = MTLTextureUsageShaderRead;
    snapDesc.storageMode = MTLStorageModeShared;
    m_mtl->displaySnapshot = [m_mtl->device newTextureWithDescriptor:snapDesc];

    // --- Vertex buffer (shared for CPU writes + GPU reads) ---
    m_mtl->vertexBuffer = [m_mtl->device newBufferWithLength:c_vertexBufferSize * sizeof(Vertex)
                                                     options:MTLResourceStorageModeShared];

    // --- Compile shaders ---
    NSError *err = nil;
    auto *lib = [m_mtl->device newLibraryWithSource:s_metalShaderSource options:nil error:&err];
    if (!lib) { NSLog(@"Metal shader compile error: %@", err); return -1; }
    auto *vertFunc = [lib newFunctionWithName:@"vertexMain"];
    auto *fragFunc = [lib newFunctionWithName:@"fragmentMain"];

    // --- Vertex descriptor ---
    auto *vd = [[MTLVertexDescriptor alloc] init];
    // attribute 0: position (int2, offset 0)
    vd.attributes[0].format      = MTLVertexFormatInt2;
    vd.attributes[0].offset      = offsetof(Vertex, x);
    vd.attributes[0].bufferIndex = 0;
    // attribute 1: color (uint, offset 8)
    vd.attributes[1].format      = MTLVertexFormatUInt;
    vd.attributes[1].offset      = offsetof(Vertex, color);
    vd.attributes[1].bufferIndex = 0;
    // attribute 2: texpage (ushort, offset 12)
    vd.attributes[2].format      = MTLVertexFormatUShort;
    vd.attributes[2].offset      = offsetof(Vertex, texpage);
    vd.attributes[2].bufferIndex = 0;
    // attribute 3: clut (ushort, offset 14)
    vd.attributes[3].format      = MTLVertexFormatUShort;
    vd.attributes[3].offset      = offsetof(Vertex, clut);
    vd.attributes[3].bufferIndex = 0;
    // attribute 4: uv (ushort2, offset 16)
    vd.attributes[4].format      = MTLVertexFormatUShort2;
    vd.attributes[4].offset      = offsetof(Vertex, u);
    vd.attributes[4].bufferIndex = 0;
    // layout
    vd.layouts[0].stride       = sizeof(Vertex);
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    // --- Pipeline states ---
    m_mtl->psoOpaque      = makePSO(m_mtl->device, vertFunc, fragFunc, vd, false, MTLBlendOperationAdd);
    m_mtl->psoBlendAdd    = makePSO(m_mtl->device, vertFunc, fragFunc, vd, true, MTLBlendOperationAdd);
    m_mtl->psoBlendRevSub = makePSO(m_mtl->device, vertFunc, fragFunc, vd, true, MTLBlendOperationReverseSubtract);

    if (!m_mtl->psoOpaque || !m_mtl->psoBlendAdd || !m_mtl->psoBlendRevSub) return -1;

    // --- RGBA8→R16Uint compute pipeline for VRAM sync ---
    {
        NSError *syncErr = nil;
        auto *syncLib = [m_mtl->device newLibraryWithSource:s_syncComputeSource options:nil error:&syncErr];
        if (!syncLib) { NSLog(@"Sync compute shader error: %@", syncErr); return -1; }
        auto *syncFunc = [syncLib newFunctionWithName:@"rgba8ToR16"];
        m_mtl->syncComputePSO = [m_mtl->device newComputePipelineStateWithFunction:syncFunc error:&syncErr];
        if (!m_mtl->syncComputePSO) { NSLog(@"Sync compute PSO error: %@", syncErr); return -1; }
    }

    // Need write access on sample texture for the compute shader
    // (already created above — but need to update usage)

    // --- Initial uniform state ---
    m_uniforms.drawOffsetX = 0.5f;
    m_uniforms.drawOffsetY = 0.5f;
    m_uniforms.texWindowMaskX = ~0;
    m_uniforms.texWindowMaskY = ~0;
    m_uniforms.texWindowOffX = 0;
    m_uniforms.texWindowOffY = 0;
    m_uniforms.blendR = m_uniforms.blendG = m_uniforms.blendB = 1.0f;
    m_uniforms.blendA = 0.0f;
    m_uniforms.blendOpaqueR = m_uniforms.blendOpaqueG = m_uniforms.blendOpaqueB = 1.0f;
    m_uniforms.blendOpaqueA = 0.0f;
    m_uniforms.setMask = 0;
    m_uniforms.checkMask = 0;

    m_gpustat = 0x14802000;
    return 0;
}

int PCSX::MetalGPU::shutdown() {
    if (m_mtl) {
        m_mtl->commitAndWait();
        delete m_mtl;
        m_mtl = nullptr;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// VRAM sync: blit render target → sample texture
// ---------------------------------------------------------------------------
void PCSX::MetalGPU::syncVRAMTexture() {
    if (!m_syncVRAM || !m_mtl) return;
    m_syncVRAM = false;

    // Only sync the current draw area (scissor region) from RT → sample texture.
    // This preserves CPU-uploaded data (CLUTs, textures) outside the draw area
    // that may not be present in the RT (e.g., wiped by FastFill).
    auto &sc = m_mtl->scissor;
    if (sc.width == 0 || sc.height == 0) return;

    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();
    auto *enc = [m_mtl->frameCmdBuf computeCommandEncoder];
    [enc setComputePipelineState:m_mtl->syncComputePSO];
    [enc setTexture:m_mtl->vramRenderTarget atIndex:0];
    [enc setTexture:m_mtl->vramSampleTexture atIndex:1];

    // Pass scissor offset to the compute shader
    uint32_t offset[2] = {(uint32_t)sc.x, (uint32_t)sc.y};
    [enc setBytes:offset length:sizeof(offset) atIndex:0];
    MTLSize gridSize = MTLSizeMake(sc.width, sc.height, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
}

// ---------------------------------------------------------------------------
// renderBatch — flush accumulated vertices to GPU
// ---------------------------------------------------------------------------
void PCSX::MetalGPU::renderBatch() {
    if (m_vertexCount <= 0 || !m_mtl) return;

    syncVRAMTexture();

    // Apply deferred draw offset
    if (m_updateDrawOffset) {
        setDrawOffset(m_lastDrawOffsetSetting);
    }

    // Upload vertices to the streaming buffer
    memcpy(m_mtl->vertexBuffer.contents, m_vertices.data(), m_vertexCount * sizeof(Vertex));

    // Select pipeline based on transparency mode
    id<MTLRenderPipelineState> pso = m_mtl->psoOpaque;
    if (m_lastTransparency == Transparency::Transparent) {
        pso = (m_lastBlendingMode == 2) ? m_mtl->psoBlendRevSub : m_mtl->psoBlendAdd;
    }

    // Encode draw into the frame command buffer (no commit, no wait)
    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();

    auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = m_mtl->vramRenderTarget;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    auto *enc = [m_mtl->frameCmdBuf renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:pso];
    [enc setVertexBuffer:m_mtl->vertexBuffer offset:0 atIndex:0];
    [enc setVertexBytes:&m_uniforms length:sizeof(m_uniforms) atIndex:1];
    [enc setFragmentBytes:&m_uniforms length:sizeof(m_uniforms) atIndex:1];
    [enc setFragmentTexture:m_mtl->vramSampleTexture atIndex:0];
    [enc setScissorRect:m_mtl->scissor];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:m_vertexCount];
    [enc endEncoding];

    s_log.totalVerts += m_vertexCount;
    s_log.batches++;
    if (m_lastTransparency == Transparency::Transparent) s_log.batchesSemiTrans++;
    else s_log.batchesOpaque++;

    // Log first batch's vertex positions once per log interval
    if (s_log.batches == 1 && (s_log.frameNum % s_log.logInterval) >= s_log.logInterval - 1) {
        int n = std::min(m_vertexCount, 6);
        fprintf(stderr, "  [batch] offset=(%.1f,%.1f) verts=%d first%d:", m_uniforms.drawOffsetX, m_uniforms.drawOffsetY, m_vertexCount, n);
        for (int i = 0; i < n; i++) fprintf(stderr, " (%d,%d tp=%04x cl=%04x)", m_vertices[i].x, m_vertices[i].y, m_vertices[i].texpage, m_vertices[i].clut);
        fprintf(stderr, "\n");
    }
    m_syncVRAM = true;  // next batch must re-sync sample texture
    m_vertexCount = 0;
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
void PCSX::MetalGPU::drawTri(int *x, int *y, uint32_t *colors) {
    maybeRenderBatch<3>();
    for (int i = 0; i < 3; i++) m_vertices[m_vertexCount++] = Vertex(x[i], y[i], colors[i]);
}

void PCSX::MetalGPU::drawTriTextured(int *x, int *y, uint32_t *colors,
                                      uint16_t clut, uint16_t texpage,
                                      unsigned *u, unsigned *v) {
    maybeRenderBatch<3>();
    for (int i = 0; i < 3; i++)
        m_vertices[m_vertexCount++] = Vertex(x[i], y[i], colors[i], clut, texpage, u[i], v[i]);
}

void PCSX::MetalGPU::drawRect(int x, int y, int w, int h, uint32_t color) {
    maybeRenderBatch<6>();
    m_vertices[m_vertexCount++] = Vertex(x,     y,     color);
    m_vertices[m_vertexCount++] = Vertex(x + w, y,     color);
    m_vertices[m_vertexCount++] = Vertex(x + w, y + h, color);
    m_vertices[m_vertexCount++] = Vertex(x + w, y + h, color);
    m_vertices[m_vertexCount++] = Vertex(x,     y + h, color);
    m_vertices[m_vertexCount++] = Vertex(x,     y,     color);
}

void PCSX::MetalGPU::drawRectTextured(int x, int y, int w, int h, uint32_t color,
                                       uint16_t clut, unsigned u, unsigned v) {
    maybeRenderBatch<6>();
    uint16_t tp = m_rectTexpage;
    m_vertices[m_vertexCount++] = Vertex(x,     y,     color, clut, tp, u,     v);
    m_vertices[m_vertexCount++] = Vertex(x + w, y,     color, clut, tp, u + w, v);
    m_vertices[m_vertexCount++] = Vertex(x + w, y + h, color, clut, tp, u + w, v + h);
    m_vertices[m_vertexCount++] = Vertex(x + w, y + h, color, clut, tp, u + w, v + h);
    m_vertices[m_vertexCount++] = Vertex(x,     y + h, color, clut, tp, u,     v + h);
    m_vertices[m_vertexCount++] = Vertex(x,     y,     color, clut, tp, u,     v);
}

void PCSX::MetalGPU::drawLine(int x1, int y1, uint32_t c1, int x2, int y2, uint32_t c2) {
    maybeRenderBatch<6>();
    const int dx = x2 - x1, dy = y2 - y1;
    if (dx == 0 && dy == 0) {
        m_vertices[m_vertexCount++] = Vertex(x1,     y1,     c1);
        m_vertices[m_vertexCount++] = Vertex(x1 + 1, y1,     c1);
        m_vertices[m_vertexCount++] = Vertex(x1 + 1, y1 + 1, c1);
        m_vertices[m_vertexCount++] = Vertex(x1 + 1, y1 + 1, c1);
        m_vertices[m_vertexCount++] = Vertex(x1,     y1 + 1, c1);
        m_vertices[m_vertexCount++] = Vertex(x1,     y1,     c1);
        return;
    }
    int xOff, yOff;
    if (std::abs(dx) > std::abs(dy)) {
        xOff = 0; yOff = 1;
        dx > 0 ? x2++ : x1++;
    } else {
        xOff = 1; yOff = 0;
        dy > 0 ? y2++ : y1++;
    }
    m_vertices[m_vertexCount++] = Vertex(x1,        y1,        c1);
    m_vertices[m_vertexCount++] = Vertex(x2,        y2,        c2);
    m_vertices[m_vertexCount++] = Vertex(x2 + xOff, y2 + yOff, c2);
    m_vertices[m_vertexCount++] = Vertex(x2 + xOff, y2 + yOff, c2);
    m_vertices[m_vertexCount++] = Vertex(x1 + xOff, y1 + yOff, c1);
    m_vertices[m_vertexCount++] = Vertex(x1,        y1,        c1);
}

// ---------------------------------------------------------------------------
// Transparency / blending state
// ---------------------------------------------------------------------------
template <PCSX::MetalGPU::Transparency setting>
void PCSX::MetalGPU::setTransparency() {
    if (m_lastTransparency == setting) return;
    renderBatch();
    m_lastTransparency = setting;
    if constexpr (setting == Transparency::Opaque) {
        m_lastBlendingMode = -1;
        setBlendFactors(1.0f, 0.0f);  // full src, no dst contribution
    }
}

void PCSX::MetalGPU::setBlendingModeFromTexpage(uint32_t texpage) {
    int mode = (texpage >> 5) & 3;
    if (m_lastBlendingMode == mode) return;
    renderBatch();
    m_lastBlendingMode = mode;
    switch (mode) {
        case 0: setBlendFactors(0.5f, 0.5f); break;   // B/2 + F/2
        case 1: setBlendFactors(1.0f, 1.0f); break;   // B + F
        case 2: setBlendFactors(1.0f, 1.0f); break;   // B - F (ReverseSubtract)
        case 3: setBlendFactors(0.25f, 1.0f); break;  // B + F/4
    }
}

void PCSX::MetalGPU::setBlendFactors(float src, float dst) {
    m_uniforms.blendR = m_uniforms.blendG = m_uniforms.blendB = src;
    m_uniforms.blendA = dst;
}

// ---------------------------------------------------------------------------
// Drawing area / offset / tex window
// ---------------------------------------------------------------------------
void PCSX::MetalGPU::updateDrawArea() {
    renderBatch();
    int left   = m_drawAreaLeft;
    int top    = m_drawAreaTop;
    int width  = std::max<int>(m_drawAreaRight - left + 1, 0);
    int height = std::max<int>(m_drawAreaBottom - top + 1, 0);
    // Clamp to VRAM bounds
    if (left + width > c_vramWidth) width = c_vramWidth - left;
    if (top + height > c_vramHeight) height = c_vramHeight - top;
    if (left < 0) { width += left; left = 0; }
    if (top < 0) { height += top; top = 0; }
    if (width < 0) width = 0;
    if (height < 0) height = 0;
    m_mtl->scissor = {(NSUInteger)left, (NSUInteger)top, (NSUInteger)width, (NSUInteger)height};
    s_log.scissorX = left; s_log.scissorY = top; s_log.scissorW = width; s_log.scissorH = height;
}

void PCSX::MetalGPU::setDrawOffset(uint32_t cmd) {
    m_updateDrawOffset = false;
    m_lastDrawOffsetSetting = cmd & 0x3fffff;
    int offsetX = (int32_t)cmd << 21 >> 21;
    int offsetY = (int32_t)cmd << 10 >> 21;
    m_drawingOffset.x() = offsetX;
    m_drawingOffset.y() = offsetY;
    m_uniforms.drawOffsetX = (float)offsetX + 0.5f;
    m_uniforms.drawOffsetY = (float)offsetY + 0.5f;
}

void PCSX::MetalGPU::setTexWindow(uint32_t cmd) {
    uint32_t setting = cmd & 0xfffff;
    if (setting == m_lastTexwindowSetting) return;
    renderBatch();
    m_lastTexwindowSetting = setting;
    uint32_t maskX = (cmd & 0x1f) * 8;
    uint32_t maskY = ((cmd >> 5) & 0x1f) * 8;
    uint32_t offsX = ((cmd >> 10) & 0x1f) * 8;
    uint32_t offsY = ((cmd >> 15) & 0x1f) * 8;
    m_uniforms.texWindowMaskX = ~maskX;
    m_uniforms.texWindowMaskY = ~maskY;
    m_uniforms.texWindowOffX  = offsX & maskX;
    m_uniforms.texWindowOffY  = offsY & maskY;
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------
uint32_t PCSX::MetalGPU::readStatusInternal() { return m_gpustat; }

void PCSX::MetalGPU::write0(ClearCache *) {}

void PCSX::MetalGPU::write0(FastFill *cmd) {
    s_log.fastFills++;
    renderBatch();
    if (!m_mtl || cmd->w == 0 || cmd->h == 0) return;

    // Convert BGR888 to RGBA8 float clear color
    uint32_t c = cmd->color;
    float r = float(c & 0xff) / 255.f;
    float g = float((c >> 8) & 0xff) / 255.f;
    float b = float((c >> 16) & 0xff) / 255.f;

    // Use a render pass with clear action on a sub-region via scissor
    auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = m_mtl->vramRenderTarget;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();

    auto *enc = [m_mtl->frameCmdBuf renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:m_mtl->psoOpaque];

    Vertex fillVerts[6];
    uint32_t col = (uint32_t)(r * 255) | ((uint32_t)(g * 255) << 8) | ((uint32_t)(b * 255) << 16);
    int fx = cmd->x, fy = cmd->y, fw = cmd->w, fh = cmd->h;
    fillVerts[0] = Vertex(fx,      fy,      col);
    fillVerts[1] = Vertex(fx + fw, fy,      col);
    fillVerts[2] = Vertex(fx + fw, fy + fh, col);
    fillVerts[3] = Vertex(fx + fw, fy + fh, col);
    fillVerts[4] = Vertex(fx,      fy + fh, col);
    fillVerts[5] = Vertex(fx,      fy,      col);

    Uniforms fillU = m_uniforms;
    fillU.drawOffsetX = 0.5f;
    fillU.drawOffsetY = 0.5f;
    fillU.blendR = fillU.blendG = fillU.blendB = 1.0f;
    fillU.blendA = 0.0f;
    fillU.blendOpaqueR = fillU.blendOpaqueG = fillU.blendOpaqueB = 1.0f;
    fillU.blendOpaqueA = 0.0f;
    fillU.setMask = 0;    // FastFill ignores mask bit
    fillU.checkMask = 0;

    [enc setVertexBytes:fillVerts length:sizeof(fillVerts) atIndex:0];
    [enc setVertexBytes:&fillU length:sizeof(fillU) atIndex:1];
    [enc setFragmentBytes:&fillU length:sizeof(fillU) atIndex:1];
    [enc setFragmentTexture:m_mtl->vramSampleTexture atIndex:0];
    MTLScissorRect fullScissor = {0, 0, (NSUInteger)c_vramWidth, (NSUInteger)c_vramHeight};
    [enc setScissorRect:fullScissor];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];

    // Update CPU-side 16-bit mirror
    uint16_t col16 = ((uint32_t)(r * 31 + 0.5f))
                   | ((uint32_t)(g * 31 + 0.5f) << 5)
                   | ((uint32_t)(b * 31 + 0.5f) << 10);
    for (unsigned row = 0; row < cmd->h; row++) {
        unsigned dy = (cmd->y + row) & 0x1ff;
        for (unsigned col = 0; col < cmd->w; col++) {
            unsigned dx = (cmd->x + col) & 0x3ff;
            m_vram16[dy * c_vramWidth + dx] = col16;
        }
    }

    // Don't write to sample texture — the scissor-scoped sync will propagate
    // the fill from the RT to the draw area of the sample. Writing here would
    // wipe CLUT/texture data stored outside the draw area.
    m_syncVRAM = true;
}

// --- Polygons ---------------------------------------------------------------
template <PCSX::GPU::Shading shading, PCSX::GPU::Shape shape, PCSX::GPU::Textured textured,
          PCSX::GPU::Blend blend, PCSX::GPU::Modulation modulation>
void PCSX::MetalGPU::polyExec(Poly<shading, shape, textured, blend, modulation> *prim) {
    if constexpr (textured == Textured::Yes) s_log.polysTex++; else s_log.polysUntex++;
    if constexpr (blend == Blend::Off) setTransparency<Transparency::Opaque>();
    else setTransparency<Transparency::Transparent>();

    if constexpr (textured == Textured::No) {
        if constexpr (blend == Blend::Semi) setBlendingModeFromTexpage(m_rectTexpage);
        drawTri(&prim->x[0], &prim->y[0], &prim->colors[0]);
        if constexpr (shape == Shape::Quad)
            drawTri(&prim->x[1], &prim->y[1], &prim->colors[1]);
    } else {
        if constexpr (blend == Blend::Semi) setBlendingModeFromTexpage(prim->tpage.raw);
        drawTriTextured(&prim->x[0], &prim->y[0], &prim->colors[0],
                        prim->clutraw, prim->tpage.raw, &prim->u[0], &prim->v[0]);
        if constexpr (shape == Shape::Quad)
            drawTriTextured(&prim->x[1], &prim->y[1], &prim->colors[1],
                            prim->clutraw, prim->tpage.raw, &prim->u[1], &prim->v[1]);
    }
}

#define POLY_STUB(s, sh, t, b, m) \
    void PCSX::MetalGPU::write0(Poly<Shading::s, Shape::sh, Textured::t, Blend::b, Modulation::m> *p) { polyExec(p); }
POLY_STUB(Flat,Tri,No,Off,Off)  POLY_STUB(Flat,Tri,No,Off,On)  POLY_STUB(Flat,Tri,No,Semi,Off) POLY_STUB(Flat,Tri,No,Semi,On)
POLY_STUB(Flat,Tri,Yes,Off,Off) POLY_STUB(Flat,Tri,Yes,Off,On) POLY_STUB(Flat,Tri,Yes,Semi,Off)POLY_STUB(Flat,Tri,Yes,Semi,On)
POLY_STUB(Flat,Quad,No,Off,Off) POLY_STUB(Flat,Quad,No,Off,On) POLY_STUB(Flat,Quad,No,Semi,Off)POLY_STUB(Flat,Quad,No,Semi,On)
POLY_STUB(Flat,Quad,Yes,Off,Off)POLY_STUB(Flat,Quad,Yes,Off,On)POLY_STUB(Flat,Quad,Yes,Semi,Off)POLY_STUB(Flat,Quad,Yes,Semi,On)
POLY_STUB(Gouraud,Tri,No,Off,Off) POLY_STUB(Gouraud,Tri,No,Off,On) POLY_STUB(Gouraud,Tri,No,Semi,Off) POLY_STUB(Gouraud,Tri,No,Semi,On)
POLY_STUB(Gouraud,Tri,Yes,Off,Off)POLY_STUB(Gouraud,Tri,Yes,Off,On)POLY_STUB(Gouraud,Tri,Yes,Semi,Off)POLY_STUB(Gouraud,Tri,Yes,Semi,On)
POLY_STUB(Gouraud,Quad,No,Off,Off) POLY_STUB(Gouraud,Quad,No,Off,On) POLY_STUB(Gouraud,Quad,No,Semi,Off) POLY_STUB(Gouraud,Quad,No,Semi,On)
POLY_STUB(Gouraud,Quad,Yes,Off,Off)POLY_STUB(Gouraud,Quad,Yes,Off,On)POLY_STUB(Gouraud,Quad,Yes,Semi,Off)POLY_STUB(Gouraud,Quad,Yes,Semi,On)
#undef POLY_STUB

// --- Lines ------------------------------------------------------------------
template <PCSX::GPU::Shading shading, PCSX::GPU::LineType lineType, PCSX::GPU::Blend blend>
void PCSX::MetalGPU::lineExec(Line<shading, lineType, blend> *prim) {
    s_log.lines++;
    if constexpr (blend == Blend::Off) setTransparency<Transparency::Opaque>();
    else setTransparency<Transparency::Transparent>();

    auto count = prim->colors.size();
    for (unsigned i = 1; i < count; i++)
        drawLine(prim->x[i - 1], prim->y[i - 1], prim->colors[i - 1],
                 prim->x[i],     prim->y[i],     prim->colors[i]);
}

#define LINE_STUB(s, lt, b) \
    void PCSX::MetalGPU::write0(Line<Shading::s, LineType::lt, Blend::b> *p) { lineExec(p); }
LINE_STUB(Flat,Simple,Off) LINE_STUB(Flat,Simple,Semi) LINE_STUB(Flat,Poly,Off) LINE_STUB(Flat,Poly,Semi)
LINE_STUB(Gouraud,Simple,Off) LINE_STUB(Gouraud,Simple,Semi) LINE_STUB(Gouraud,Poly,Off) LINE_STUB(Gouraud,Poly,Semi)
#undef LINE_STUB

// --- Rectangles -------------------------------------------------------------
template <PCSX::GPU::Size size, PCSX::GPU::Textured textured,
          PCSX::GPU::Blend blend, PCSX::GPU::Modulation modulation>
void PCSX::MetalGPU::rectExec(Rect<size, textured, blend, modulation> *prim) {
    if constexpr (textured == Textured::Yes) s_log.rectsTex++; else s_log.rectsUntex++;
    if constexpr (blend == Blend::Off) setTransparency<Transparency::Opaque>();
    else {
        setTransparency<Transparency::Transparent>();
        setBlendingModeFromTexpage(m_rectTexpage);
    }

    if constexpr (textured == Textured::No) {
        drawRect(prim->x, prim->y, prim->w, prim->h, prim->color);
    } else {
        uint32_t color = 0x808080;
        if constexpr (modulation == Modulation::On) color = prim->color;
        drawRectTextured(prim->x, prim->y, prim->w, prim->h, color, prim->clutraw, prim->u, prim->v);
    }
}

#define RECT_STUB(sz, t, b, m) \
    void PCSX::MetalGPU::write0(Rect<Size::sz, Textured::t, Blend::b, Modulation::m> *p) { rectExec(p); }
RECT_STUB(Variable,No,Off,Off) RECT_STUB(Variable,No,Semi,Off) RECT_STUB(Variable,Yes,Off,Off) RECT_STUB(Variable,Yes,Semi,Off)
RECT_STUB(S1,No,Off,Off) RECT_STUB(S1,No,Semi,Off) RECT_STUB(S1,Yes,Off,Off) RECT_STUB(S1,Yes,Semi,Off)
RECT_STUB(S8,No,Off,Off) RECT_STUB(S8,No,Semi,Off) RECT_STUB(S8,Yes,Off,Off) RECT_STUB(S8,Yes,Semi,Off)
RECT_STUB(S16,No,Off,Off) RECT_STUB(S16,No,Semi,Off) RECT_STUB(S16,Yes,Off,Off) RECT_STUB(S16,Yes,Semi,Off)
RECT_STUB(Variable,No,Off,On) RECT_STUB(Variable,No,Semi,On) RECT_STUB(Variable,Yes,Off,On) RECT_STUB(Variable,Yes,Semi,On)
RECT_STUB(S1,No,Off,On) RECT_STUB(S1,No,Semi,On) RECT_STUB(S1,Yes,Off,On) RECT_STUB(S1,Yes,Semi,On)
RECT_STUB(S8,No,Off,On) RECT_STUB(S8,No,Semi,On) RECT_STUB(S8,Yes,Off,On) RECT_STUB(S8,Yes,Semi,On)
RECT_STUB(S16,No,Off,On) RECT_STUB(S16,No,Semi,On) RECT_STUB(S16,Yes,Off,On) RECT_STUB(S16,Yes,Semi,On)
#undef RECT_STUB

// --- VRAM blits -------------------------------------------------------------
void PCSX::MetalGPU::write0(BlitVramVram *cmd) {
    s_log.blitVV++;
    renderBatch();
    if (!m_mtl || cmd->w == 0 || cmd->h == 0) return;

    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();
    auto *blit = [m_mtl->frameCmdBuf blitCommandEncoder];
    [blit copyFromTexture:m_mtl->vramRenderTarget
              sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(cmd->sX, cmd->sY, 0)
               sourceSize:MTLSizeMake(cmd->w, cmd->h, 1)
                toTexture:m_mtl->vramRenderTarget
         destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(cmd->dX, cmd->dY, 0)];
    [blit endEncoding];
    m_syncVRAM = true;
}

// --- State commands ---------------------------------------------------------
void PCSX::MetalGPU::write0(TPage *cmd) { m_rectTexpage = cmd->raw & 0x1ff; }
void PCSX::MetalGPU::write0(TWindow *cmd) { setTexWindow(cmd->raw); }
void PCSX::MetalGPU::write0(DrawingAreaStart *cmd) { m_drawAreaLeft = cmd->x; m_drawAreaTop = cmd->y; updateDrawArea(); }
void PCSX::MetalGPU::write0(DrawingAreaEnd *cmd) { m_drawAreaRight = cmd->x; m_drawAreaBottom = cmd->y; updateDrawArea(); }
void PCSX::MetalGPU::write0(DrawingOffset *cmd) {
    renderBatch();
    uint32_t word = cmd->raw & 0x3fffff;
    if (word != m_lastDrawOffsetSetting) { m_updateDrawOffset = true; m_lastDrawOffsetSetting = word; }
}
void PCSX::MetalGPU::write0(MaskBit *cmd) {
    renderBatch();
    m_uniforms.setMask = cmd->set ? 1 : 0;
    m_uniforms.checkMask = cmd->check ? 1 : 0;
}

// --- Control commands -------------------------------------------------------
void PCSX::MetalGPU::write1(CtrlReset *) { resetBackend(); }
void PCSX::MetalGPU::write1(CtrlClearFifo *) {}
void PCSX::MetalGPU::write1(CtrlIrqAck *) {}
void PCSX::MetalGPU::write1(CtrlDisplayEnable *cmd) { m_display.enabled = cmd->enable; }
void PCSX::MetalGPU::write1(CtrlDmaSetting *cmd) {
    m_gpustat = (m_gpustat & ~0x60000000) | (static_cast<uint32_t>(cmd->dma) << 29);
}
void PCSX::MetalGPU::write1(CtrlDisplayStart *cmd) { m_display.set(cmd); }
void PCSX::MetalGPU::write1(CtrlHorizontalDisplayRange *cmd) { m_display.set(cmd); }
void PCSX::MetalGPU::write1(CtrlVerticalDisplayRange *cmd) { m_display.set(cmd); }
void PCSX::MetalGPU::write1(CtrlDisplayMode *cmd) { m_display.set(cmd); }

// ---------------------------------------------------------------------------
// vblank — present VRAM to GUI offscreen texture
// ---------------------------------------------------------------------------
void* PCSX::MetalGPU::getDisplayTextureID() {
    if (!m_mtl) return nullptr;
    // Always return displayTexture — it gets populated every vblank
    // either from 24-bit decode or from a readback of the render target.
    return (__bridge void*)m_mtl->displayTexture;
}

void PCSX::MetalGPU::getDisplayUVs(float &u0, float &v0, float &u1, float &v1) {
    if (m_mtl && m_mtl->display24bit) {
        u0 = 0.0f; v0 = 0.0f;
        u1 = float(m_display.size.x()) / float(c_vramWidth);
        v1 = float(m_display.size.y()) / float(c_vramHeight);
    } else {
        // Use raw start/size integers to compute UVs directly,
        // avoiding potential issues with startNormalized not being updated.
        u0 = float(m_display.start.x()) / float(c_vramWidth);
        v0 = float(m_display.start.y()) / float(c_vramHeight);
        u1 = u0 + float(m_display.size.x()) / float(c_vramWidth);
        v1 = v0 + float(m_display.size.y()) / float(c_vramHeight);
    }
}

void PCSX::MetalGPU::vblank(bool fromGui) {
    renderBatch();
    s_log.dump(m_display);
    if (!m_mtl) return;

    // Commit the frame's command buffer — ONE GPU sync per frame.
    m_mtl->commitAndWait();

    m_mtl->display24bit = m_display.enabled && (m_display.info.depth == CtrlDisplayMode::CD_24BITS);

    if (m_mtl->display24bit) {
        // 24-bit display decode from raw 16-bit VRAM mirror
        int dispX = m_display.start.x();
        int dispY = m_display.start.y();
        int dispW = m_display.size.x();
        int dispH = m_display.size.y();
        if (dispW <= 0 || dispH <= 0) return;
        if (dispW > c_vramWidth) dispW = c_vramWidth;
        if (dispH > c_vramHeight) dispH = c_vramHeight;

        std::vector<uint32_t> decoded(dispW * dispH);
        for (int y = 0; y < dispH; y++) {
            int srcY = (dispY + y) & 511;
            for (int x = 0; x < dispW; x++) {
                int byteOff = x * 3;
                int wordIdx = byteOff / 2;
                int byteInWord = byteOff & 1;
                int srcX = (dispX + wordIdx) & 1023;
                int srcX1 = (srcX + 1) & 1023;

                uint16_t w0 = m_vram16[srcY * c_vramWidth + srcX];
                uint16_t w1 = m_vram16[srcY * c_vramWidth + srcX1];

                uint8_t rb, gb, bb;
                if (byteInWord == 0) {
                    rb = w0 & 0xff;
                    gb = (w0 >> 8) & 0xff;
                    bb = w1 & 0xff;
                } else {
                    rb = (w0 >> 8) & 0xff;
                    gb = w1 & 0xff;
                    bb = (w1 >> 8) & 0xff;
                }
                decoded[y * dispW + x] = rb | (gb << 8) | (bb << 16) | 0xff000000u;
            }
        }
        [m_mtl->displayTexture replaceRegion:MTLRegionMake2D(0, 0, dispW, dispH)
                                 mipmapLevel:0
                                   withBytes:decoded.data()
                                 bytesPerRow:dispW * 4];
    } else {
        // Blit VRAM render target → display texture for ImGui.
        // The frame command buffer was already committed above, so the render target
        // has all the frame's draws. Use a new command buffer for the readback.
        constexpr size_t vramBytes = c_vramWidth * c_vramHeight * 4;
        auto *staging = [m_mtl->device newBufferWithLength:vramBytes options:MTLResourceStorageModeShared];
        auto *cb = [m_mtl->commandQueue commandBuffer];
        auto *blit = [cb blitCommandEncoder];
        [blit copyFromTexture:m_mtl->vramRenderTarget sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(c_vramWidth, c_vramHeight, 1)
                   toBuffer:staging destinationOffset:0 destinationBytesPerRow:c_vramWidth * 4
          destinationBytesPerImage:vramBytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // Force alpha=1 so ImGui renders all pixels as fully opaque.
        // The RT alpha stores PSX MSB (mask bit state) which is not display alpha.
        auto *px = static_cast<uint8_t *>(staging.contents);
        for (int i = 0; i < c_vramWidth * c_vramHeight; i++) px[i * 4 + 3] = 0xff;

        [m_mtl->displayTexture replaceRegion:MTLRegionMake2D(0, 0, c_vramWidth, c_vramHeight)
                                 mipmapLevel:0
                                   withBytes:staging.contents
                                 bytesPerRow:c_vramWidth * 4];
    }
}


// ---------------------------------------------------------------------------
// VRAM access
// ---------------------------------------------------------------------------
void PCSX::MetalGPU::clearVRAM() {
    if (!m_mtl) return;
    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();
    auto *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = m_mtl->vramRenderTarget;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0, 0, 0, 0);
    auto *enc = [m_mtl->frameCmdBuf renderCommandEncoderWithDescriptor:rpd];
    [enc endEncoding];
    m_syncVRAM = true;
}

void PCSX::MetalGPU::resetBackend() {
    m_gpustat = 0x14802000;
    m_vertexCount = 0;
    m_lastBlendingMode = -1;
    m_lastTransparency = Transparency::Opaque;
    m_lastTexwindowSetting = 0;
    m_lastDrawOffsetSetting = 0;
    m_updateDrawOffset = false;
    m_syncVRAM = true;
    m_uniforms.drawOffsetX = 0.5f;
    m_uniforms.drawOffsetY = 0.5f;
    clearVRAM();
}

GLuint PCSX::MetalGPU::getVRAMTexture() { return 0; }
void PCSX::MetalGPU::setLinearFiltering() {}

PCSX::Slice PCSX::MetalGPU::getVRAM(Ownership) {
    if (!m_mtl) return {};
    renderBatch();
    m_mtl->commitAndWait();  // ensure all GPU work is done before readback

    constexpr size_t vramBytes = c_vramWidth * c_vramHeight * 4;  // RGBA8
    auto *staging = [m_mtl->device newBufferWithLength:vramBytes options:MTLResourceStorageModeShared];

    auto *cmdBuf = [m_mtl->commandQueue commandBuffer];
    auto *blit = [cmdBuf blitCommandEncoder];
    [blit copyFromTexture:m_mtl->vramRenderTarget
              sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(c_vramWidth, c_vramHeight, 1)
                 toBuffer:staging destinationOffset:0 destinationBytesPerRow:c_vramWidth * 4
      destinationBytesPerImage:vramBytes];
    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Convert RGBA8 → 16-bit (1555)
    constexpr size_t pixCount = c_vramWidth * c_vramHeight;
    auto *rgba = (const uint8_t *)staging.contents;
    auto *out16 = new uint16_t[pixCount];
    for (size_t i = 0; i < pixCount; i++) {
        uint8_t r = rgba[i * 4 + 0], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2], a = rgba[i * 4 + 3];
        out16[i] = ((r * 31 + 127) / 255) | (((g * 31 + 127) / 255) << 5) | (((b * 31 + 127) / 255) << 10) | ((a >> 7) << 15);
    }

    Slice slice;
    slice.acquire(std::string(reinterpret_cast<const char *>(out16), pixCount * 2));
    delete[] out16;
    return slice;
}

void PCSX::MetalGPU::partialUpdateVRAM(int x, int y, int w, int h, const uint16_t *pixels, PartialUpdateVram) {
    s_log.cpuUploads++;
    if (!m_mtl || w <= 0 || h <= 0) return;

    renderBatch();

    // Update CPU-side raw 16-bit mirror (needed for 24-bit display decode)
    for (int row = 0; row < h; row++) {
        int destY = (y + row) & 0x1ff;
        for (int col = 0; col < w; col++) {
            int destX = (x + col) & 0x3ff;
            m_vram16[destY * c_vramWidth + destX] = pixels[row * w + col];
        }
    }

    // Must commit pending GPU work before CPU writes to shared textures
    m_mtl->commitAndWait();

    // Upload raw 16-bit directly to sample texture (R16Uint) — no conversion!
    [m_mtl->vramSampleTexture replaceRegion:MTLRegionMake2D(x, y, w, h)
                                mipmapLevel:0
                                  withBytes:pixels
                                bytesPerRow:w * sizeof(uint16_t)];

    // Also upload RGBA8 version to render target (for display readback)
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
    auto *staging = [m_mtl->device newBufferWithBytes:rgba.data() length:byteSize options:MTLResourceStorageModeShared];
    m_mtl->ensureNoRenderEncoder();
    m_mtl->ensureFrameCmdBuf();
    auto *blit = [m_mtl->frameCmdBuf blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:w * 4 sourceBytesPerImage:byteSize
               sourceSize:MTLSizeMake(w, h, 1)
               toTexture:m_mtl->vramRenderTarget destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(x, y, 0)];
    [blit endEncoding];

    m_syncVRAM = true;
}

bool PCSX::MetalGPU::configure() { return false; }
void PCSX::MetalGPU::debug() {}

#endif  // PCSX_METAL
