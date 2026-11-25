#pragma once
#include <cuda_runtime.h>
#include <cmath>

#define HD __host__ __device__ inline

HD float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HD float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


HD float3 operator*(float3 a, float s){
    return make_float3(a.x * s, a.y * s, a.z * s);
}

HD void operator+=(float3& a, float3 b){
    a = a + b;
}

HD void operator-=(float3& a, float3 b){
    a = a - b;
}

HD float dot(float3 a, float3 b){
    return((a.x * b.x) + (a.y * b.y)  + (a.z * b.z));
}

HD float length(float3 a){
    return sqrt(dot(a, a));
}

HD float3 cross(float3 a, float3 b){
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

HD float3 normalize(float3 a){
    return(a * (1.0f / length(a)));
}
#undef HD
