#pragma once
#include "screen.cuh"
#include <SDL.h>
#include <SDL_image.h>

// Same idea as CUDA_CHECK: stop with SDL's messages when a call fails.

#define SDL_CHECK(ok)                                               \
    do{                                                             \
        if (!(ok)){                                                 \
            std::fprintf(stderr, "SDL error %s at %s:%d\n",         \
                         SDL_GetError(), __FILE__, __LINE__);       \
            std::exit(1);                                           \
        }                                                           \
    }while(0)

struct Window{
    SDL_Window*   win; //the os window
    SDL_Renderer* ren; // draws into the window
    SDL_Texture*  tex; //streaming, RGBA32, same size as the Screen
};

static Window winOpen(int W, int H){
    Window w{};
    SDL_CHECK(SDL_Init(SDL_INIT_VIDEO) == 0);
    w.win = SDL_CreateWindow("koi", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, 0);
    SDL_CHECK(w.win);
    w.ren = SDL_CreateRenderer(w.win, -1, 0);   // flags 0: no vsync, the loop paces itself
    SDL_CHECK(w.ren);
    w.tex = SDL_CreateTexture(w.ren, SDL_PIXELFORMAT_RGBA32,
                              SDL_TEXTUREACCESS_STREAMING, W, H);
    SDL_CHECK(w.tex);

    SDL_RendererInfo ri;
    SDL_GetRendererInfo(w.ren, &ri);
    std::printf("video driver %s, renderer %s, window %dx%d\n",
                SDL_GetCurrentVideoDriver(), ri.name, W, H);
    return w;
}

// Host copy -> texture -> window. The pitch is the byte length of one row.

static void winShow(Window& w, const Screen& sc){
    SDL_UpdateTexture(w.tex, nullptr, sc.host, sc.W * (int)sizeof(uchar4));
    SDL_RenderCopy(w.ren, w.tex, nullptr, nullptr);
    SDL_RenderPresent(w.ren);
}

static void winClose(Window& w){
    SDL_DestroyTexture(w.tex);
    SDL_DestroyRenderer(w.ren);
    SDL_DestroyWindow(w.win);
    SDL_Quit();
}

// What the keys changed. Step 7b adds the mouse camera here.

struct UI{
    bool quit = false, paused = false, reseed = false, shot = false;
    unsigned seed = 1; // the seed of the pond on screen
    int shots = 0;     // screenshots saved so far
};

// Read every waiting event into ui. Key repeat is ignored: one action per press.
static void pollEvents(UI& ui){
    SDL_Event e;
    while(SDL_PollEvent(&e)){
        if(e.type == SDL_QUIT){
            ui.quit = true;
        }
        if(e.type != SDL_KEYDOWN || e.key.repeat){
            continue;
        }
        switch(e.key.keysym.sym){
        case SDLK_ESCAPE: ui.quit = true;                  break;
        case SDLK_SPACE:  ui.paused = !ui.paused;          break;
        case SDLK_r:      ui.seed++;  ui.reseed = true;    break;
        case SDLK_s:      ui.shot = true;                  break;
        }
    }
}

// S key: save the frame on screen as shotNNN.png in the current folder.
static void winSave(const Screen& sc, int n){
    char name[32];
    std::snprintf(name, sizeof name, "shot%03d.png", n);
    SDL_Surface* img = SDL_CreateRGBSurfaceWithFormatFrom(
        sc.host, sc.W, sc.H, 32, sc.W * (int)sizeof(uchar4), SDL_PIXELFORMAT_RGBA32);
    SDL_CHECK(img);
    if(IMG_SavePNG(img, name) == 0){
        uchar4 p = sc.host[sc.W - 1];
        std::printf("saved %s, top right pixel %d %d %d %d\n", name, p.x, p.y, p.z, p.w);
    }
    else{
        std::printf("screenshot failed: %s\n", SDL_GetError());
    }
    SDL_FreeSurface(img);
}
