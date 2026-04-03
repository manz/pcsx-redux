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

std::unique_ptr<PCSX::GPU> PCSX::GPU::getMetal() { return std::make_unique<PCSX::MetalGPU>(); }

PCSX::MetalGPU::MetalGPU() {}

PCSX::MetalGPU::~MetalGPU() { shutdown(); }

int PCSX::MetalGPU::initBackend(UI *) {
    m_device = MTLCreateSystemDefaultDevice();
    if (!m_device) {
        g_system->message("Metal GPU: Failed to create MTLDevice\n");
        return -1;
    }

    m_commandQueue = [m_device newCommandQueue];
    if (!m_commandQueue) {
        g_system->message("Metal GPU: Failed to create command queue\n");
        return -1;
    }

    // Create VRAM texture (1024x512, RGBA8)
    MTLTextureDescriptor *vramDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                       width:c_vramWidth
                                                                                      height:c_vramHeight
                                                                                   mipmapped:NO];
    vramDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    vramDesc.storageMode = MTLStorageModePrivate;
    m_vramTexture = [m_device newTextureWithDescriptor:vramDesc];
    if (!m_vramTexture) {
        g_system->message("Metal GPU: Failed to create VRAM texture\n");
        return -1;
    }

    m_vram16.fill(0);
    m_gpustat = 0x14802000;

    return 0;
}

int PCSX::MetalGPU::shutdown() {
    m_vramTexture = nil;
    m_commandQueue = nil;
    m_device = nil;
    return 0;
}

uint32_t PCSX::MetalGPU::readStatusInternal() { return m_gpustat; }

void PCSX::MetalGPU::vblank(bool fromGui) {
    // TODO: present VRAM contents to display
}

bool PCSX::MetalGPU::configure() {
    // TODO: Metal GPU configuration UI
    return false;
}

void PCSX::MetalGPU::debug() {
    // TODO: Metal GPU debug UI
}

void PCSX::MetalGPU::clearVRAM() {
    m_vram16.fill(0);
    // TODO: clear Metal VRAM texture
}

void PCSX::MetalGPU::resetBackend() {
    m_gpustat = 0x14802000;
    clearVRAM();
}

GLuint PCSX::MetalGPU::getVRAMTexture() {
    // Metal path does not use OpenGL textures.
    // VRAM viewers that need the texture should use getVRAMMetalTexture() instead.
    return 0;
}

void PCSX::MetalGPU::setLinearFiltering() {
    // TODO: configure Metal sampler for linear vs nearest
}

PCSX::Slice PCSX::MetalGPU::getVRAM(Ownership ownership) {
    constexpr size_t vramSize = c_vramWidth * c_vramHeight * sizeof(uint16_t);

    if (ownership == Ownership::ACQUIRE) {
        auto *copy = new uint16_t[c_vramWidth * c_vramHeight];
        std::memcpy(copy, m_vram16.data(), vramSize);
        Slice slice;
        slice.acquire(std::string(reinterpret_cast<const char *>(copy), vramSize));
        delete[] copy;
        return slice;
    }

    Slice slice;
    slice.borrow(m_vram16.data(), vramSize);
    return slice;
}

void PCSX::MetalGPU::partialUpdateVRAM(int x, int y, int w, int h, const uint16_t *pixels, PartialUpdateVram) {
    for (int row = 0; row < h; row++) {
        int destY = (y + row) & 0x1ff;
        for (int col = 0; col < w; col++) {
            int destX = (x + col) & 0x3ff;
            m_vram16[destY * c_vramWidth + destX] = pixels[row * w + col];
        }
    }
    // TODO: upload changed region to Metal VRAM texture
}

// ---------------------------------------------------------------------------
// Drawing command stubs
// ---------------------------------------------------------------------------

void PCSX::MetalGPU::write0(ClearCache *) {}

void PCSX::MetalGPU::write0(FastFill *cmd) {
    // Fill VRAM region with solid color
    uint16_t color16 = ((cmd->color >> 3) & 0x1f) | (((cmd->color >> 11) & 0x1f) << 5) |
                        (((cmd->color >> 19) & 0x1f) << 10);
    for (unsigned row = 0; row < cmd->h; row++) {
        unsigned destY = (cmd->y + row) & 0x1ff;
        for (unsigned col = 0; col < cmd->w; col++) {
            unsigned destX = (cmd->x + col) & 0x3ff;
            m_vram16[destY * c_vramWidth + destX] = color16;
        }
    }
    // TODO: encode Metal blit to clear VRAM texture region
}

// Polygon stubs -- all route through polyExec
template <PCSX::GPU::Shading shading, PCSX::GPU::Shape shape, PCSX::GPU::Textured textured, PCSX::GPU::Blend blend,
          PCSX::GPU::Modulation modulation>
void PCSX::MetalGPU::polyExec(Poly<shading, shape, textured, blend, modulation> *) {
    // TODO: encode Metal render command for polygon
}

#define POLY_STUB(s, sh, t, b, m)                                                        \
    void PCSX::MetalGPU::write0(Poly<Shading::s, Shape::sh, Textured::t, Blend::b, Modulation::m> *p) { \
        polyExec(p);                                                                      \
    }

POLY_STUB(Flat, Tri, No, Off, Off)
POLY_STUB(Flat, Tri, No, Off, On)
POLY_STUB(Flat, Tri, No, Semi, Off)
POLY_STUB(Flat, Tri, No, Semi, On)
POLY_STUB(Flat, Tri, Yes, Off, Off)
POLY_STUB(Flat, Tri, Yes, Off, On)
POLY_STUB(Flat, Tri, Yes, Semi, Off)
POLY_STUB(Flat, Tri, Yes, Semi, On)
POLY_STUB(Flat, Quad, No, Off, Off)
POLY_STUB(Flat, Quad, No, Off, On)
POLY_STUB(Flat, Quad, No, Semi, Off)
POLY_STUB(Flat, Quad, No, Semi, On)
POLY_STUB(Flat, Quad, Yes, Off, Off)
POLY_STUB(Flat, Quad, Yes, Off, On)
POLY_STUB(Flat, Quad, Yes, Semi, Off)
POLY_STUB(Flat, Quad, Yes, Semi, On)
POLY_STUB(Gouraud, Tri, No, Off, Off)
POLY_STUB(Gouraud, Tri, No, Off, On)
POLY_STUB(Gouraud, Tri, No, Semi, Off)
POLY_STUB(Gouraud, Tri, No, Semi, On)
POLY_STUB(Gouraud, Tri, Yes, Off, Off)
POLY_STUB(Gouraud, Tri, Yes, Off, On)
POLY_STUB(Gouraud, Tri, Yes, Semi, Off)
POLY_STUB(Gouraud, Tri, Yes, Semi, On)
POLY_STUB(Gouraud, Quad, No, Off, Off)
POLY_STUB(Gouraud, Quad, No, Off, On)
POLY_STUB(Gouraud, Quad, No, Semi, Off)
POLY_STUB(Gouraud, Quad, No, Semi, On)
POLY_STUB(Gouraud, Quad, Yes, Off, Off)
POLY_STUB(Gouraud, Quad, Yes, Off, On)
POLY_STUB(Gouraud, Quad, Yes, Semi, Off)
POLY_STUB(Gouraud, Quad, Yes, Semi, On)

#undef POLY_STUB

// Line stubs -- all route through lineExec
template <PCSX::GPU::Shading shading, PCSX::GPU::LineType lineType, PCSX::GPU::Blend blend>
void PCSX::MetalGPU::lineExec(Line<shading, lineType, blend> *) {
    // TODO: encode Metal render command for line
}

#define LINE_STUB(s, lt, b)                                                   \
    void PCSX::MetalGPU::write0(Line<Shading::s, LineType::lt, Blend::b> *p) { \
        lineExec(p);                                                          \
    }

LINE_STUB(Flat, Simple, Off)
LINE_STUB(Flat, Simple, Semi)
LINE_STUB(Flat, Poly, Off)
LINE_STUB(Flat, Poly, Semi)
LINE_STUB(Gouraud, Simple, Off)
LINE_STUB(Gouraud, Simple, Semi)
LINE_STUB(Gouraud, Poly, Off)
LINE_STUB(Gouraud, Poly, Semi)

#undef LINE_STUB

// Rect stubs -- all route through rectExec
template <PCSX::GPU::Size size, PCSX::GPU::Textured textured, PCSX::GPU::Blend blend, PCSX::GPU::Modulation modulation>
void PCSX::MetalGPU::rectExec(Rect<size, textured, blend, modulation> *) {
    // TODO: encode Metal render command for rectangle
}

#define RECT_STUB(sz, t, b, m)                                                          \
    void PCSX::MetalGPU::write0(Rect<Size::sz, Textured::t, Blend::b, Modulation::m> *p) { \
        rectExec(p);                                                                    \
    }

RECT_STUB(Variable, No, Off, Off)
RECT_STUB(Variable, No, Semi, Off)
RECT_STUB(Variable, Yes, Off, Off)
RECT_STUB(Variable, Yes, Semi, Off)
RECT_STUB(S1, No, Off, Off)
RECT_STUB(S1, No, Semi, Off)
RECT_STUB(S1, Yes, Off, Off)
RECT_STUB(S1, Yes, Semi, Off)
RECT_STUB(S8, No, Off, Off)
RECT_STUB(S8, No, Semi, Off)
RECT_STUB(S8, Yes, Off, Off)
RECT_STUB(S8, Yes, Semi, Off)
RECT_STUB(S16, No, Off, Off)
RECT_STUB(S16, No, Semi, Off)
RECT_STUB(S16, Yes, Off, Off)
RECT_STUB(S16, Yes, Semi, Off)
RECT_STUB(Variable, No, Off, On)
RECT_STUB(Variable, No, Semi, On)
RECT_STUB(Variable, Yes, Off, On)
RECT_STUB(Variable, Yes, Semi, On)
RECT_STUB(S1, No, Off, On)
RECT_STUB(S1, No, Semi, On)
RECT_STUB(S1, Yes, Off, On)
RECT_STUB(S1, Yes, Semi, On)
RECT_STUB(S8, No, Off, On)
RECT_STUB(S8, No, Semi, On)
RECT_STUB(S8, Yes, Off, On)
RECT_STUB(S8, Yes, Semi, On)
RECT_STUB(S16, No, Off, On)
RECT_STUB(S16, No, Semi, On)
RECT_STUB(S16, Yes, Off, On)
RECT_STUB(S16, Yes, Semi, On)

#undef RECT_STUB

void PCSX::MetalGPU::write0(BlitVramVram *cmd) {
    // CPU-side VRAM blit
    for (unsigned row = 0; row < cmd->h; row++) {
        unsigned srcY = (cmd->sY + row) & 0x1ff;
        unsigned dstY = (cmd->dY + row) & 0x1ff;
        for (unsigned col = 0; col < cmd->w; col++) {
            unsigned srcX = (cmd->sX + col) & 0x3ff;
            unsigned dstX = (cmd->dX + col) & 0x3ff;
            m_vram16[dstY * c_vramWidth + dstX] = m_vram16[srcY * c_vramWidth + srcX];
        }
    }
    // TODO: encode Metal blit
}

// State commands
void PCSX::MetalGPU::write0(TPage *) {
    // TODO: track texture page state
}

void PCSX::MetalGPU::write0(TWindow *) {
    // TODO: track texture window state
}

void PCSX::MetalGPU::write0(DrawingAreaStart *cmd) {
    m_drawAreaLeft = cmd->x;
    m_drawAreaTop = cmd->y;
}

void PCSX::MetalGPU::write0(DrawingAreaEnd *cmd) {
    m_drawAreaRight = cmd->x;
    m_drawAreaBottom = cmd->y;
}

void PCSX::MetalGPU::write0(DrawingOffset *cmd) {
    m_drawingOffset.x() = cmd->x;
    m_drawingOffset.y() = cmd->y;
}

void PCSX::MetalGPU::write0(MaskBit *) {
    // TODO: track mask bit state
}

// Control commands
void PCSX::MetalGPU::write1(CtrlReset *) {
    resetBackend();
}

void PCSX::MetalGPU::write1(CtrlClearFifo *) {}

void PCSX::MetalGPU::write1(CtrlIrqAck *) {}

void PCSX::MetalGPU::write1(CtrlDisplayEnable *cmd) {
    m_display.enabled = cmd->enable;
}

void PCSX::MetalGPU::write1(CtrlDmaSetting *cmd) {
    m_gpustat = (m_gpustat & ~0x60000000) | (static_cast<uint32_t>(cmd->dma) << 29);
}

void PCSX::MetalGPU::write1(CtrlDisplayStart *cmd) {
    m_display.set(cmd);
}

void PCSX::MetalGPU::write1(CtrlHorizontalDisplayRange *cmd) {
    m_display.set(cmd);
}

void PCSX::MetalGPU::write1(CtrlVerticalDisplayRange *cmd) {
    m_display.set(cmd);
}

void PCSX::MetalGPU::write1(CtrlDisplayMode *cmd) {
    m_display.set(cmd);
}

#endif  // PCSX_METAL
