#include "sim.cuh"
#include <cstring>

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
    }
}

int main(int argc, char** argv){
    int nFish = DEFAULT_FISH;
    bool bench = false;

    for(int i = 1; i < argc; i++){
        if(!std::strcmp(argv[i], "--bench")){
            bench = true;
        }
        else if(!std::strcmp(argv[i], "--fish")&& i + 1 < argc){
            nFish = std::atoi(argv[++i]);
        }
        else{
            std::printf("usage: %s [--fish N] [--bench]\n", argv[0]);
            return 1;
        }
    }
    if (nFish < 1) nFish =1;

    cudaDeviceProp gpu;
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s\n", gpu.name);
    Sim s{};
    simInit(s, nFish, make_float3(TANK_X, TANK_Y, TANK_Z));
    simSeed(s, 1u);
    std::printf("%d fish, tank %.0fx%.0fx%.0f, grid %dx%dx%d = %d cells\n",
                s.N, s.tank.x, s.tank.y, s.tank.z,
                s.grid.x, s.grid.y, s.grid.z, s.nCells);

    simStep(s, 1.0f / 60.0f);    //temp
    CUDA_CHECK(cudaDeviceSynchronize());//temp
    std::printf("one step ok, bench %d\n", (int)bench); //temp
    simFree(s);
    return 0;
}