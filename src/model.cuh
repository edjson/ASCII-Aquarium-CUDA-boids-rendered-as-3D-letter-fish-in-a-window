#pragma once
#include "config.cuh"
#include "vec3.cuh"
#include <vector>

// One point of the koi's skin. V runs around the boyd; 0 and 1 on the belly,
// 0.5 on the back. On a fin, v runs across the fin's rays instead.

struct KoiPoint{
    float3 p;
    float3 n;
    float u, v;
    int part;
};

enum{
     P_BODY, P_TAIL, P_DORSAL, P_PEC, P_EYE, P_BARBEL, NPART 
};

// Body radius at u, as a fraction of the shoulder's: a round head (a quarter
// circle up to the shoulder), then a cosine taper down to the tail stem.

// circle up to the shoulder), then a cosine taper down to the tail stem.
static float bodyR(float u){
    u = fminf(fmaxf(u, 0.0f), U_STEM);
    if(u < U_SHO){
        float t = 1.0f - u / U_SHO;
        return sqrtf(1.0f - t * t);
    }
    float s = (u - U_SHO) / (U_STEM - U_SHO);
    return STEM_R + (1.0f - STEM_R) * 0.5f * (1.0f + cosf(3.14159265f * s));
}

// The skin point at (u, v): an ellipse around the body axis.
static float3 bodyAt(float u, float v){
    float a = 6.2831853f * v, r = bodyR(u);
    return make_float3((0.5f - u) * KOI_LEN, -cosf(a) * r * BODY_H, sinf(a) * r * BODY_W);
}