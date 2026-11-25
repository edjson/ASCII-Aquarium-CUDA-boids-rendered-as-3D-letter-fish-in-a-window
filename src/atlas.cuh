#pragma once
#include "window.cuh"                           // SDL and SDL_CHECK
#include <SDL_ttf.h>
#include <vector>

// Draw each glyph once with SDL_ttf and keep its coverage (how much of each
// pixel the letter covers, 0..255): atlas[(g * CH + y) * CW + x] for glyph g.
// No window needed, so a headless run can bake it too.
static std::vector<unsigned char> atlasBake(){
    SDL_CHECK(TTF_Init() == 0);
    TTF_Font* font = TTF_OpenFont(FONT_FILE, FONT_PT);
    SDL_CHECK(font);                            // a wrong FONT_FILE stops here
    std::vector<unsigned char> a(NGLYPH * CH * CW, 0);

    for(int g = 0; g < NGLYPH; g++){
        SDL_Surface* s = TTF_RenderGlyph_Blended(font, GLYPHS[g], SDL_Color{255, 255, 255, 255});
        SDL_CHECK(s);
        for(int y = 0; y < CH && FONT_Y0 + y < s->h; y++){
            // Rows are s->pitch bytes apart, and the pitch can be more than w * 4.
            const Uint32* row = (const Uint32*)((const Uint8*)s->pixels + (FONT_Y0 + y) * s->pitch);
            for(int x = 0; x < CW && x < s->w; x++){
                a[(g * CH + y) * CW + x] = row[x] >> 24;   // ARGB8888: alpha is the top byte
            }
        }
        SDL_FreeSurface(s);
    }
    TTF_CloseFont(font);
    TTF_Quit();
    return a;
}