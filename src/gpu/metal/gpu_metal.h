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

#pragma once

#ifdef PCSX_METAL

#include <array>
#include <cstdint>
#include <vector>

#include "core/gpu.h"

namespace PCSX {

class MetalGPU final : public GPU {
  public:
    MetalGPU();
    ~MetalGPU() override;

    void *getDisplayTextureID() override;
    void getDisplayUVs(float &u0, float &v0, float &u1, float &v1) override;

  private:
    int initBackend(UI *) override;
    int shutdown() override;
    uint32_t readStatusInternal() override;
    void vblank(bool fromGui) override;
    bool configure() override;
    void debug() override;

    void setDither(int setting) override { m_useDither = setting; }
    void setCachedDithering(bool cached) override {}
    void clearVRAM() override;
    void resetBackend() override;
    GLuint getVRAMTexture() override;
    void setLinearFiltering() override;
    Slice getVRAM(Ownership) override;
    void partialUpdateVRAM(int x, int y, int w, int h, const uint16_t *pixels, PartialUpdateVram) override;
    void restoreStatus(uint32_t status) override { m_gpustat = status; }

    void write0(ClearCache *) override;
    void write0(FastFill *) override;

    template <Shading shading, Shape shape, Textured textured, Blend blend, Modulation modulation>
    void polyExec(Poly<shading, shape, textured, blend, modulation> *);
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Tri, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Flat, Shape::Quad, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Tri, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Poly<Shading::Gouraud, Shape::Quad, Textured::Yes, Blend::Semi, Modulation::On> *) override;

    template <Shading shading, LineType lineType, Blend blend>
    void lineExec(Line<shading, lineType, blend> *);
    void write0(Line<Shading::Flat, LineType::Simple, Blend::Off> *) override;
    void write0(Line<Shading::Flat, LineType::Simple, Blend::Semi> *) override;
    void write0(Line<Shading::Flat, LineType::Poly, Blend::Off> *) override;
    void write0(Line<Shading::Flat, LineType::Poly, Blend::Semi> *) override;
    void write0(Line<Shading::Gouraud, LineType::Simple, Blend::Off> *) override;
    void write0(Line<Shading::Gouraud, LineType::Simple, Blend::Semi> *) override;
    void write0(Line<Shading::Gouraud, LineType::Poly, Blend::Off> *) override;
    void write0(Line<Shading::Gouraud, LineType::Poly, Blend::Semi> *) override;

    template <Size size, Textured textured, Blend blend, Modulation modulation>
    void rectExec(Rect<size, textured, blend, modulation> *);
    void write0(Rect<Size::Variable, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::Variable, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::Variable, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::Variable, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S1, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S1, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S1, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S1, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S8, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S8, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S8, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S8, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S16, Textured::No, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S16, Textured::No, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::S16, Textured::Yes, Blend::Off, Modulation::Off> *) override;
    void write0(Rect<Size::S16, Textured::Yes, Blend::Semi, Modulation::Off> *) override;
    void write0(Rect<Size::Variable, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::Variable, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::Variable, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::Variable, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S1, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S1, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S1, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S1, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S8, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S8, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S8, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S8, Textured::Yes, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S16, Textured::No, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S16, Textured::No, Blend::Semi, Modulation::On> *) override;
    void write0(Rect<Size::S16, Textured::Yes, Blend::Off, Modulation::On> *) override;
    void write0(Rect<Size::S16, Textured::Yes, Blend::Semi, Modulation::On> *) override;

    void write0(BlitVramVram *) override;

    void write0(TPage *) override;
    void write0(TWindow *) override;
    void write0(DrawingAreaStart *) override;
    void write0(DrawingAreaEnd *) override;
    void write0(DrawingOffset *) override;
    void write0(MaskBit *) override;

    void write1(CtrlReset *) override;
    void write1(CtrlClearFifo *) override;
    void write1(CtrlIrqAck *) override;
    void write1(CtrlDisplayEnable *) override;
    void write1(CtrlDmaSetting *) override;
    void write1(CtrlDisplayStart *) override;
    void write1(CtrlHorizontalDisplayRange *) override;
    void write1(CtrlVerticalDisplayRange *) override;
    void write1(CtrlDisplayMode *) override;

    // -----------------------------------------------------------------------
    // Vertex format — matches the OpenGL GPU layout exactly (20 bytes)
    // -----------------------------------------------------------------------
    struct Vertex {
        int32_t x, y;          // position (signed 11-bit range)
        uint32_t color;        // BGR888 (top 8 bits unused)
        uint16_t texpage;      // texture page word, bit 15 = untextured
        uint16_t clut;         // CLUT address
        uint16_t u, v;         // texture coordinates

        static constexpr uint16_t c_untexturedTexpage = 0x8000;

        Vertex() = default;
        Vertex(int px, int py, uint32_t col)
            : x(px), y(py), color(col), texpage(c_untexturedTexpage), clut(0), u(0), v(0) {}
        Vertex(int px, int py, uint32_t col, uint16_t cl, uint16_t tp, unsigned tu, unsigned tv)
            : x(px), y(py), color(col), texpage(tp), clut(cl), u(tu), v(tv) {}
    };

    // -----------------------------------------------------------------------
    // Uniforms passed to vertex + fragment shaders
    // -----------------------------------------------------------------------
    struct Uniforms {
        float drawOffsetX, drawOffsetY;
        int32_t texWindowMaskX, texWindowMaskY, texWindowOffX, texWindowOffY;
        float blendR, blendG, blendB, blendA;           // per-primitive blend factors
        float blendOpaqueR, blendOpaqueG, blendOpaqueB, blendOpaqueA;
        int32_t setMask;    // force MSB=1 on written pixels
        int32_t checkMask;  // skip writes to pixels with MSB=1
    };

    // -----------------------------------------------------------------------
    // Rendering helpers
    // -----------------------------------------------------------------------
    static constexpr int c_vramWidth = 1024;
    static constexpr int c_vramHeight = 512;
    static constexpr int c_vertexBufferSize = 0x100000;

    enum class Transparency { Opaque, Transparent };

    template <int count>
    void maybeRenderBatch() {
        if ((m_vertexCount + count) >= c_vertexBufferSize) renderBatch();
    }
    void renderBatch();
    void syncVRAMTexture();

    void drawTri(int *x, int *y, uint32_t *colors);
    void drawTriTextured(int *x, int *y, uint32_t *colors, uint16_t clut, uint16_t texpage, unsigned *u, unsigned *v);
    void drawRect(int x, int y, int w, int h, uint32_t color);
    void drawRectTextured(int x, int y, int w, int h, uint32_t color, uint16_t clut, unsigned u, unsigned v);
    void drawLine(int x1, int y1, uint32_t c1, int x2, int y2, uint32_t c2);

    template <Transparency setting>
    void setTransparency();
    void setBlendingModeFromTexpage(uint32_t texpage);
    void setBlendFactors(float src, float dst);
    void updateDrawArea();
    void setDrawOffset(uint32_t cmd);
    void setTexWindow(uint32_t cmd);

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    uint32_t m_gpustat = 0x14802000;
    int m_useDither = 0;

    std::vector<Vertex> m_vertices;
    int m_vertexCount = 0;
    Uniforms m_uniforms{};

    int m_drawAreaLeft = 0, m_drawAreaRight = 0;
    int m_drawAreaTop = 0, m_drawAreaBottom = 0;
    OpenGL::ivec2 m_drawingOffset;

    Transparency m_lastTransparency = Transparency::Opaque;
    int m_lastBlendingMode = -1;
    uint32_t m_rectTexpage = 0;
    uint32_t m_lastTexwindowSetting = 0;
    uint32_t m_lastDrawOffsetSetting = 0;
    bool m_updateDrawOffset = false;
    bool m_syncVRAM = true;

    // CPU-side raw 16-bit VRAM mirror — needed for 24-bit display decode
    // where raw byte data must be preserved (RGBA8 expansion loses byte boundaries)
    std::array<uint16_t, 1024 * 512> m_vram16{};

    // Opaque pointer to Objective-C Metal objects — defined in gpu_metal.mm
    struct MetalObjects;
    MetalObjects *m_mtl = nullptr;
};

}  // namespace PCSX

#endif  // PCSX_METAL
