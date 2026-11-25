#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                         cudaGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(1);                                               \
        }                                                               \
    } while (0)


// World scale: one koi is 10 units long. Pond = 10 x 3 x 8 koi lengths.
// y is up and shallow; x and z are the surface.
constexpr float TANK_X = 100.0f;
constexpr float TANK_Y = 30.0f;
constexpr float TANK_Z = 80.0f;

// Boids, tuned for koi: loose, slow, drifting groups. L = one koi length.
constexpr float VIEW_R    = 14.0f;  // 1.4 L: neighbor radius, also the cell size
constexpr float SEP_R     = 10.0f;  // 1.0 L (was 5): koi keep more space
constexpr float SEP_W     = 20.0f;  // push at zero distance, fades to 0 at SEP_R
constexpr float ALI_W     = 0.5f;   // was 8: loose heading match
constexpr float COH_W     = 0.15f;  // was 3: gentle pull to the local center
constexpr float WANDER_W  = 3.5f;   // each koi's own slow nudge
constexpr float MAX_FORCE = 6.0f;   // steering cap: gentle turns
constexpr float SPEED     = 10.0f;  // top speed, 1.0 L/s (was 30)
constexpr float CRUISE    = 7.0f;   // speed a koi settles back to, 0.7 L/s
constexpr float MIN_SPEED = 4.0f;   // never stall, 0.4 L/s
constexpr float CRUISE_W  = 1.5f;   // how fast it eases back to level cruising
constexpr float MAX_CLIMB = 0.5f;   // steepest climb: vertical / horizontal speed
constexpr float WALL_M    = 20.0f;  // soft wall margin on x and z (2 L)
constexpr float WALL_MY   = 8.0f;   // soft wall margin on y (the pond is shallow)
constexpr float WALL_W    = 35.0f;  // soft wall push right at the wall
constexpr int   DEFAULT_FISH = 64;  // --fish N overrides it
// Character cells. DejaVu Sans Mono at 13 pt draws each glyph below in an
// 8 x 16 box whose rows 0 and 1 are always empty, so a cell keeps rows 2-15.
constexpr int  CW = 8, CH = 14;              // cell width and height in pixels
constexpr int  FONT_PT = 13, FONT_Y0 = 2;    // font size (1 pt = 1 px), first row kept
constexpr char FONT_FILE[] = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";

// Atlas order. 0-9: brightness ramp, darkest to brightest by measured ink.
// 10-14: edges / | \ - _ ('-' is in both sets). 15: bubble.
constexpr char GLYPHS[] = " .-:*+=%#@/|\\-_o";
constexpr int  NGLYPH = 16, NRAMP = 10, G_EDGE = 10, G_BUBBLE = 15;
static_assert(sizeof(GLYPHS) == NGLYPH + 1, "one character per glyph");
