#include <SDL.h>
#include <zlib.h>

#include "core/psxcommon.h"
#include "core/r3000a.h"
#include "gui/gui.h"

void SysPrintf(const char *fmt, ...) {
    // print message to debugging console
    va_list a;
    va_start(a, fmt);
    vprintf(fmt, a);
    va_end(a);
}

void SysMessage(const char *fmt, ...) {
    // display message to user as a pop-up
    va_list a;
    va_start(a, fmt);
    vprintf(fmt, a);
    va_end(a);
}

void SysUpdate() {
    // called on vblank to update states
    GUI_flip();
}

void SysRunGui() {
    // called when the UI needs to show up
}

void SysReset() {
    // debugger is requesting a reset
}

void SysClose() {
    // emulator is requesting a shutdown of the emulation
}

int main() {
    unsigned int texture = GUI_init();

    memset(&g_config, 0, sizeof(PcsxConfig));
    g_config.PsxAuto = 1;
    g_config.HLE = 0;
    g_config.SlowBoot = 0;
    strcpy(g_config.BiosDir, ".");
    strcpy(g_config.Bios, "bios.bin");

    SetIsoFile("test.img");
    LoadPlugins();

    GPU_open(texture);

    EmuInit();
    EmuReset();

    CDR_open();
    CheckCdrom();
    LoadCdrom();

    g_psxCpu = &g_psxInt;
    g_psxCpu->Init();
    g_psxCpu->Execute();

    // temporary, to make sure the code doesn't get removed at link time
    g_psxCpu = &g_psxRec;
    g_psxCpu->Init();
    g_psxCpu->Execute();

    return 0;
}