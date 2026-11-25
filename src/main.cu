#include "sim.cuh"
#include <cstring>
#include "screen.cuh"
#include "window.cuh"
#include "atlas.cuh"
#include "fish.cuh"

// 500 headless frames: time the sim, see how much of the pond is used,
// and follow every fish by its id.
static void runBench(Sim& s){
    const int FRAMES = 500;
    const float dt   = 1.0f / 60.0f;
    std::vector<float3> hp(s.N), last(s.N);   // positions now / last frame, by id
    std::vector<int>    hid(s.N);
    float3 lo = make_float3( 1e30f,  1e30f,  1e30f);
    float3 hi = make_float3(-1e30f, -1e30f, -1e30f);
    float maxStep = 0.0f, totalMs = 0.0f;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for(int f = 0; f < FRAMES; f++){
        CUDA_CHECK(cudaEventRecord(t0));
        simStep(s, dt);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1)); // wait until the GPU finished the step
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        totalMs += ms;
        CUDA_CHECK(cudaMemcpy(hp.data(), s.pos, s.N * sizeof(float3),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hid.data(), s.fishId, s.N * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for(int k = 0; k < s.N; k++){
            float3 p = hp[k];
            lo.x = fminf(lo.x, p.x); hi.x = fmaxf(hi.x, p.x);
            lo.y = fminf(lo.y, p.y); hi.y = fmaxf(hi.y, p.y);
            lo.z = fminf(lo.z, p.z); hi.z = fmaxf(hi.z, p.z);
            int id = hid[k];
            if(id < 0 || id >= s.N){
                std::printf("bad fishId %d in slot %d\n", id, k);
                std::exit(1);
            }
            if(f > 0){
                maxStep = fmaxf(maxStep, length(p - last[id]));
            }
            last[id] = p;
        }
    }
    std::printf("sim %.3f ms/frame (avg of %d)\n", totalMs / FRAMES, FRAMES);
    std::printf("x  %6.2f .. %6.2f   seeded %.0f .. %.0f\n",
                lo.x, hi.x, WALL_M, s.tank.x - WALL_M);
    std::printf("y  %6.2f .. %6.2f   seeded %.0f .. %.0f\n",
                lo.y, hi.y, WALL_MY, s.tank.y - WALL_MY);
    std::printf("z  %6.2f .. %6.2f   seeded %.0f .. %.0f\n",
                lo.z, hi.z, WALL_M, s.tank.z - WALL_M);
    std::printf("max step %.4f   limit %.4f (SPEED * dt)\n", maxStep, SPEED * dt);
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

// the window: events, sim step, draw, copy to the host, show, 60 times a second.

// Everything the GPU draws in one frame, in order. 
static void drawFrame(const Screen& scr, const Camera& cam, const Model& koi,
                      const unsigned char* atlas){
    k_gather_cells<<<blocksFor(scr.cols, scr.rows), TILE>>>(scr, cam, koi.pts, koi.n);
    k_rasterize<<<blocksFor(scr.W, scr.H), TILE>>>(scr, atlas);
    CUDA_CHECK(cudaGetLastError());
}

static void runWindow(Sim& s, int W, int H){
    unsigned char* atlas = atlasUpload(atlasBake());
    Window win = winOpen(W, H);

    Screen scr = screenInit(W, H);
    std::printf("atlas %d glyphs of %dx%d px, %dx%d cells\n", NGLYPH, CW, CH, scr.cols, scr.rows);
    std::vector<KoiPoint> points = modelBuild(72, 64);   // the koi, built once on the host
    modelPrint(points);
    Model koi = modelUpload(points);
    Patterns pat = patternInit(6);  //temp
    patternSeed(pat, 6, 1u);
    UI   ui;                       
    const float dt = 1.0f / 60.0f;             // one bench per frame 60 fps
    Uint64 hz = SDL_GetPerformanceFrequency(); // counter ticks per secound
    Uint64 due = SDL_GetPerformanceCounter(); // when the next frame is due
    Uint64 tFps = due;      // when the fps count started
    int frames = 0;         // frames since tFps
    double workMs = 0.0;   // time spent working (not sleeping) since tFps

    while (!ui.quit){
        Uint64 t0 = SDL_GetPerformanceCounter();
        pollEvents(ui);
        if(ui.reseed){
            simSeed(s, ui.seed);
            patternSeed(pat, 6, ui.seed);
            ui.reseed = false;
        }
        if(!ui.paused){
            simStep(s, dt);
        }
        // TEMP 
        Camera cam = camOrbit(make_float3(0.0f, 0.0f, 0.0f), 0.4f * s.time, 0.35f, 100.0f, 44.0f);
        drawFrame(scr, cam, koi, atlas);

        screenDownload(scr); //waits for GPU
        if(ui.shot){
            winSave(scr, ui.shots++);
            ui.shot = false;
        }
        winShow(win, scr);

        // 60 fps caps: sleep until the next frame is due, 1/60 s after the last one.
        Uint64 now = SDL_GetPerformanceCounter();
        due += hz / 60;
        if(now < due){
            SDL_Delay((Uint32)((due - now) * 1000 / hz));
        }
        else{
            due = now;
        }
        // once a second: put fps and work time in the title bar
        workMs += (now - t0) * 1000.0 / hz;
        frames++;
        double sec = (now - tFps) / (double)hz;
        if(sec >= 1.0){
            char title[96];
            std::snprintf(title, sizeof title, "koi  %d fish  seed %u  %.0f fps  %.2f ms%s",
                          s.N, ui.seed, frames / sec, workMs / frames, ui.paused ? "  paused" : "");
            SDL_SetWindowTitle(win.win, title);
            frames = 0;
            workMs = 0.0;
            tFps = now;
        }  
    }
    screenFree(scr);
    cudaFree(atlas);
    cudaFree(koi.pts);
    patternFree(pat);
    winClose(win);  
}

int main(int argc, char** argv){
    int nFish = DEFAULT_FISH;
    int W = 1280, H = 960;                          // --size WxH
    bool bench = false, showAtlas = false;

    for(int i = 1; i < argc; i++){
        if(!std::strcmp(argv[i], "--bench")){
            bench = true;
        }
        else if(!std::strcmp(argv[i], "--atlas")){
            showAtlas = true;
        }
        else if(!std::strcmp(argv[i], "--fish")&& i + 1 < argc){
            nFish = std::atoi(argv[++i]);
        }
        else if(!std::strcmp(argv[i], "--size") && i + 1 < argc &&
                std::sscanf(argv[i + 1], "%dx%d", &W, &H) == 2 && W >= 64 && H >= 64){
                i++;
        }
        else{
            std::printf("usage: %s [--fish N] [--bench] [--size WxH] [--atlas]\n", argv[0]);
            return 1;
        }
    }
    if(nFish < 1){
        nFish =1;
    }
    if(showAtlas){
        atlasPrint(atlasBake());
        return 0;
    }

    cudaDeviceProp gpu;
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s\n", gpu.name);
    Sim s{};
    simInit(s, nFish, make_float3(TANK_X, TANK_Y, TANK_Z));
    simSeed(s, 1u);
    std::printf("%d fish, tank %.0fx%.0fx%.0f, grid %dx%dx%d = %d cells\n",
                s.N, s.tank.x, s.tank.y, s.tank.z,
                s.grid.x, s.grid.y, s.grid.z, s.nCells);
    if(bench){
        runBench(s);
    }
    else{
        runWindow(s, W, H);
    }
    simFree(s);
    return 0;
}
