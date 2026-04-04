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

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

#include "gui/gui_metal.h"

// Global device pointer for widgets that need to create Metal textures
id<MTLDevice> g_metalDevice = nil;

namespace PCSX {
namespace MetalUI {

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    CAMetalLayer* layer = nil;
    MTLRenderPassDescriptor* renderPassDescriptor = nil;
    GLFWwindow* window = nullptr;
};

void* getDevice() { return (__bridge void*)g_metalDevice; }

void* init(GLFWwindow* window) {
    auto* ctx = new MetalContext();
    ctx->window = window;

    ctx->device = MTLCreateSystemDefaultDevice();
    g_metalDevice = ctx->device;
    ctx->commandQueue = [ctx->device newCommandQueue];

    NSWindow* nswin = glfwGetCocoaWindow(window);
    ctx->layer = [CAMetalLayer layer];
    ctx->layer.device = ctx->device;
    ctx->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    ctx->layer.contentsScale = nswin.backingScaleFactor;
    nswin.contentView.layer = ctx->layer;
    nswin.contentView.wantsLayer = YES;

    ctx->renderPassDescriptor = [MTLRenderPassDescriptor new];

    // Pre-configure the render pass with a dummy texture so ImGui_ImplMetal_NewFrame
    // can read the sample count before the first frame's drawable is available.
    MTLTextureDescriptor *dummyDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:1 height:1 mipmapped:NO];
    dummyDesc.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> dummyTexture = [ctx->device newTextureWithDescriptor:dummyDesc];
    ctx->renderPassDescriptor.colorAttachments[0].texture = dummyTexture;

    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(ctx->device);

    return ctx;
}

void shutdown(void* opaque) {
    auto* ctx = static_cast<MetalContext*>(opaque);
    if (!ctx) return;
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ctx->renderPassDescriptor = nil;
    ctx->layer = nil;
    ctx->commandQueue = nil;
    ctx->device = nil;
    delete ctx;
}

void newFrame(void* opaque) {
    auto* ctx = static_cast<MetalContext*>(opaque);
    ImGui_ImplMetal_NewFrame(ctx->renderPassDescriptor);
}

void renderFrame(void* opaque, ImDrawData* drawData, float clearR, float clearG, float clearB, float clearA) {
    auto* ctx = static_cast<MetalContext*>(opaque);
    @autoreleasepool {
        int fbW, fbH;
        glfwGetFramebufferSize(ctx->window, &fbW, &fbH);

        ctx->layer.drawableSize = CGSizeMake(fbW, fbH);
        id<CAMetalDrawable> drawable = [ctx->layer nextDrawable];
        if (!drawable) return;

        ctx->renderPassDescriptor.colorAttachments[0].clearColor =
            MTLClearColorMake(clearR * clearA, clearG * clearA, clearB * clearA, clearA);
        ctx->renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        ctx->renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        ctx->renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:ctx->renderPassDescriptor];

        ImGui_ImplMetal_RenderDrawData(drawData, commandBuffer, renderEncoder);

        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:drawable];
        [commandBuffer commit];
    }
}

void destroyFontsTexture(void*) {
    ImGui_ImplMetal_DestroyFontsTexture();
}

void createFontsTexture(void* opaque) {
    auto* ctx = static_cast<MetalContext*>(opaque);
    ImGui_ImplMetal_CreateFontsTexture(ctx->device);
}

}  // namespace MetalUI
}  // namespace PCSX

#endif  // PCSX_METAL
