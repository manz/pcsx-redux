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

struct GLFWwindow;
struct ImDrawData;

namespace PCSX {
namespace MetalUI {

void* init(GLFWwindow* window);
void shutdown(void* ctx);
void newFrame(void* ctx);
void renderFrame(void* ctx, ImDrawData* drawData, float clearR, float clearG, float clearB, float clearA);
void destroyFontsTexture(void* ctx);
void createFontsTexture(void* ctx);

}  // namespace MetalUI
}  // namespace PCSX

#endif  // PCSX_METAL
