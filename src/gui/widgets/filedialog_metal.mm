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
#include <cstdint>
#include <vector>

extern id<MTLDevice> g_metalDevice;

extern "C" void* metalCreateTexture(uint8_t* data, int w, int h, char fmt) {
    if (!g_metalDevice) return nullptr;
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                          width:w height:h mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [g_metalDevice newTextureWithDescriptor:desc];
    if (fmt == 0) {
        // BGRA → RGBA swizzle
        std::vector<uint8_t> rgba(w * h * 4);
        for (int i = 0; i < w * h; i++) {
            rgba[i * 4 + 0] = data[i * 4 + 2];
            rgba[i * 4 + 1] = data[i * 4 + 1];
            rgba[i * 4 + 2] = data[i * 4 + 0];
            rgba[i * 4 + 3] = data[i * 4 + 3];
        }
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:rgba.data() bytesPerRow:w * 4];
    } else {
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data bytesPerRow:w * 4];
    }
    return (__bridge_retained void*)tex;
}

extern "C" void metalDeleteTexture(void* tex) {
    if (tex) {
        id<MTLTexture> mtlTex = (__bridge_transfer id<MTLTexture>)tex;
        mtlTex = nil;
    }
}

#endif  // PCSX_METAL
