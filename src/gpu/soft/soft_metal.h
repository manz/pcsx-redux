/*
 * Metal display bridge for the software GPU renderer.
 * C++-compatible functions callable from .cc files.
 */

#pragma once

#ifdef PCSX_METAL

#include <cstdint>

struct SoftMetalDisplay;

SoftMetalDisplay *softMetalCreateDisplay(int width, int height);
void softMetalDestroyDisplay(SoftMetalDisplay *display);
void softMetalUpload16(SoftMetalDisplay *display, const uint16_t *vram16);
void softMetalUpload24(SoftMetalDisplay *display, const uint8_t *vram, int byteOffset);
void *softMetalGetTextureID(SoftMetalDisplay *display);

#endif  // PCSX_METAL
