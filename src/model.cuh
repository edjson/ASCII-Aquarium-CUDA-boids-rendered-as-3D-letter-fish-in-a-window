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

struct Model{
    KoiPoint *pts;   //device, n points
    int       n;
};

static Model modelUpload(const std::vector<KoiPoint>& m){
    Model d{nullptr, (int)m.size()};
    CUDA_CHECK(cudaMalloc(&d.pts, m.size() * sizeof(KoiPoint)));
    CUDA_CHECK(cudaMemcpy(d.pts, m.data(), m.size() * sizeof(KoiPoint), cudaMemcpyHostToDevice));
    return d;
}

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
            float u = U_STEM * (i + 0.5f) / rings, v = (j + 0.5f) / around;
            m.push_back(KoiPoint{bodyAt(u, v), bodyNormal(u, v), u, v, P_BODY});
        }
    }
    // Everthing else is added through add(): the point's u follows from its
    // x, so a fin will swing with the part of the body it sints on.

    auto add = [&](float3 p, float3 n, float v, int part){
        m.push_back(KoiPoint{p, n, 0.5f - (p.x / KOI_LEN), v, part});
    };
    const float3 SIDE = make_float3(0.0f, 0.0f, 1.0f); // normal of an upright fin
    
    // Tail fin: an upright fan from the stem to two tips, with a notch cut into
    // its trailing edge. t runs along the fin, s from the bottom (-1) to the top (1).
    for(int i = 0; i < 24; i++){
        for(int j = 0; j < 24; j++){
            float t = (i + 0.5f) / 24, s = (j + 0.5f) / 24 * 2.0f - 1.0f;
            if(fabsf(s) < 0.9f * (t - 0.45f)){
                continue;
            }
            float x = -0.25f * KOI_LEN * (1.0f + t);
            float h = 0.44f + 1.26f * powf(t, 0.8f);
            add(make_float3(x, s * h, 0.0f), SIDE, 0.5f + 0.5f * s, P_TAIL);
        }
    }

    // dorsal fins: long and low on the back, tallest near its front.
    for(int i = 0; i < 40; i++){
        for(int j = 0; j < 5; j++){
            float t = (i + 0.5f) / 40, s = (j + 0.5f) / 5;
            float u = 0.36f + 0.32f * t;
            float h = 0.5f * fminf(t / 0.12f, 1.0f) * (1.0f - 0.6f * t);
            float y = bodyR(u) * BODY_H - 0.05f + s * h; // from just under the skin upward
            add(make_float3((0.5f - u) * KOI_LEN, y, 0.0f), SIDE, t, P_DORSAL);
        }
    }

    // Pectoral fins: two flat fans on the sides behind the head, swept back
    // and tilted a little downward. side = -1, then +1.
    for(int side = -1; side <= 1; side += 2){
        float3 root = bodyAt(0.24f, 0.5f - side * 0.29f);
        float3 back = make_float3(-1.0f, 0.0f, 0.0f);
        float3 out  = make_float3(0.0f, -0.17f, side * 0.985f);     // outward, a little down
        float3 n    = cross(back, out) * (float)side;               // the fan's upper face
        for(int i = 0; i < 12; i++){
            for(int j = 0; j < 12; j++){
                float s   = (i + 0.5f) / 12, w = (j + 0.5f) / 12;
                float a   = 0.45f + 0.80f * w;                              // angle from straight back
                float len = 1.9f * (0.65f + 0.35f * sinf(3.14159265f * w)); // a rounded tip
                add(root + (back * cosf(a) + out * sinf(a)) * (s * len), n, w, P_PEC);
            }
        }
    }
    // Eyes: one point each, lifted 0.25 off the skin so that it is the nearest
    // point of its cell. Barbels: two short whiskers at the corners of the mouth.
    for(int side = -1; side <= 1; side += 2){
        float  v = 0.5f - side * 0.17f;
        float3 n = bodyNormal(0.075f, v);
        add(bodyAt(0.075f, v) + n * 0.25f, n, v, P_EYE);
        float3 base = bodyAt(0.02f, 0.5f - side * 0.36f);
        float3 dir  = normalize(make_float3(0.5f, -0.6f, side * 0.6f));
        for(int k = 0; k < 6; k++){
            add(base + dir * (0.5f * (k + 0.5f) / 6), dir, 0.0f, P_BARBEL);
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