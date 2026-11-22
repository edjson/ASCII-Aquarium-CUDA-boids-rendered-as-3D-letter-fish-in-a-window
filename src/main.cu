#include "sim.cuh"
#include <cstring>

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
    // TEMP vec3 test: Block 1v deletes it
    float3 a = make_float3(1, 2, 3), b = make_float3(10, 20, 40);
    float3 c = a + b, d = b - a, e = a * 2.0f;
    c += a;  d -= a;
    std::printf("%g %g %g | %g %g %g | %g %g %g | %g %g\n", c.x, c.y, c.z,
                d.x, d.y, d.z, e.x, e.y, e.z, dot(a, b), length(make_float3(2, 3, 6)));

    std::printf("%d fish, bench %d\n", nFish, (int)bench);
    return 0;
}