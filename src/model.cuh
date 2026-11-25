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

static float3 bodyNormal(float u, float v){
    const float E = 0.002f; 
    float3 du = bodyAt(u + E, v) - bodyAt(u - E, v);
    float3 dv = bodyAt(u, v + E) - bodyAt(u, v - E);
    return normalize(cross(dv, du));
}

// Build the koi once on the host: a few loops over the profile functions.
// The body gets rings x around points; the fins have fixed counts.
static std::vector<KoiPoint> modelBuild(int rings, int around){
    std::vector<KoiPoint> m;
    
    // Body: rings along the axis, around points on each ring.
    for(int i = 0; i < rings; i++){
        for(int j = 0; j < around; j++){
            float u = U_STEM * (i + 0.5f) / rings, v = (j + 0.05f) / around;
            m.push_back(KoiPoint{bodyAt(u, v), bodyNormal(u, v), u, v, P_BODY});
        }
    }
    return m;
}

// Counts and bounds: compare these digits with the l lesson's 
static void modelPrint(const std::vector<KoiPoint>& m){
    int count[NPART] = {};
    float3 lo = m[0].p, hi = m[0].p;
    for(const KoiPoint& k : m){
        count[k.part]++;
        lo.x = fminf(lo.x, k.p.x); hi.x = fmaxf(hi.x, k.p.x);
        lo.y = fminf(lo.y, k.p.y); hi.y = fmaxf(hi.y, k.p.y);
        lo.z = fminf(lo.z, k.p.z); hi.z = fmaxf(hi.z, k.p.z);
    }
    std::printf("koi: %zu points (body %d, tail %d, dorsal %d, pectoral %d, eye %d, barbel %d)\n",
                m.size(), count[P_BODY], count[P_TAIL], count[P_DORSAL], count[P_PEC], count[P_EYE],
                count[P_BARBEL]);
        std::printf("x %.2f .. %.2f   y %.2f .. %.2f   z %.2f .. %.2f\n",
                lo.x, hi.x, lo.y, hi.y, lo.z, hi.z);
}