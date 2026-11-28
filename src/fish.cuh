#pragma once
#include "screen.cuh"
#include "pattern.cuh"

// The camera: where it stands and its three axes, each of length 1. zoom is
// the scale in pixels per world unit. light points from the scene toward the
// light; it is fixed in the world and rides along, so a kernel that gets the
// camera has the light too. Step 7b rebuilds the camera from the mouse.
struct Camera{
    float3 eye, right, up, fwd;
    float  zoom;
    float3 light;
};

// A camera that looks at target from dist away. yaw walks it around the
// target, pitch lifts it above (radians; with pitch > 0 it looks down).
static Camera camOrbit(float3 target, float yaw, float pitch, float dist, float zoom){
    Camera c;
    c.fwd   = make_float3(-sinf(yaw) * cosf(pitch), -sinf(pitch), cosf(yaw) * cosf(pitch));
    c.eye   = target - c.fwd * dist;
    c.right = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), c.fwd));
    c.up    = cross(c.fwd, c.right);
    c.zoom  = zoom;
    c.light = normalize(make_float3(-0.6f, 1.0f, -0.6f));   // from the upper left, a bit in front
    return c;
}

// Where does the world point P land? Three dot products with the camera's
// axes say how far away, how far right and how far up it is. From those its
// pixel, and from the pixel its character cell (cx, cy). z is the depth along
// the view. Returns false for a point off the screen.
__device__ inline bool toCell(Screen sc, Camera cam, float3 P, int& cx, int& cy, float& z){
    float3 d = P - cam.eye;
    z = dot(d, cam.fwd);
      if(z < 1.0f){                                            // at the camera or behind it
        return false;
    }
    float s  = cam.zoom / z;                                 // twice as far, half the size
    float sx = (sc.W * 0.5f) + (s * dot(d, cam.right));      // pixel x
    float sy = (sc.H * 0.5f) - (s * dot(d, cam.up));         // pixel y, growing downward
    if(sx < 0.0f || sy < 0.0f){                              // before the casts: (int) rounds to 0
        return false;
    }
    cx = (int)sx / CW;
    cy = (int)sy / CH;
    return cx < sc.cols && cy < sc.rows;
}

// The letter for a brightness: glyphs 1 to 9 of the atlas, the ramp without
// its blank, so a covered cell always shows a letter. The square root keeps
// dim skin on the heavier letters; the cell's color carries the brightness.
__device__ inline int rampGlyph(float shade){
    return min(max((int)(sqrtf(shade) * NRAMP), 1), NRAMP - 1);
}

// The four pattern colors, in the order of the C_ names: off-white, gold,
// near-black (a dark blue-grey, so it shows on the dark water), red.
__device__ inline float3 koiPaint(int color){
    const float3 PAINT[4] = {make_float3(0.92f, 0.90f, 0.84f), make_float3(1.00f, 0.66f, 0.12f),
                             make_float3(0.18f, 0.21f, 0.31f), make_float3(0.96f, 0.27f, 0.10f)};
    return PAINT[color];
}

// One cell's depth word, 64 bits. The depth sits in the high half, so of two
// words the smaller one is the nearer point. What that point looks like rides
// along in the low half:
//   bits 63..32  depth: the float's own bits (positive floats order like integers)
//   bits 31..16  shade: 0..65535 stands for a brightness of 0..2
//   bits 15..12  part     bits 11..8  pattern color     bits 7..0  fish id, its low 8 bits
__host__ __device__ inline unsigned long long packCell(unsigned depthBits, float shade, int part,
                                                       int color, int id){
    unsigned lo = ((unsigned)(fminf(shade * 0.5f, 1.0f) * 65535.0f) << 16)
                | ((unsigned)part << 12) | ((unsigned)color << 8) | ((unsigned)id & 255u);
    return ((unsigned long long)depthBits << 32) | lo;
}


// The way back: one field of a depth word each. Only packCell and these four
// know the layout.
__host__ __device__ inline float cellShade(unsigned long long w){
    return ((unsigned)w >> 16) * (2.0f / 65535.0f);
}
__host__ __device__ inline int cellPart(unsigned long long w){
    return ((unsigned)w >> 12) & 15;
}
__host__ __device__ inline int cellColor(unsigned long long w){
    return ((unsigned)w >> 8) & 15;
}
__host__ __device__ inline int cellFish(unsigned long long w){
    return (unsigned)w & 255;
}

__device__ inline float cellDepth(unsigned long long w){
    return __uint_as_float((unsigned)(w >> 32));         // the high half, as the float it was
}

// Each koi's tail beat moves on with its speed: one beat per STRIDE koi
// lengths travelled. The phase belongs to the koi, so fishId finds it.
__global__ void k_phase(const float3* vel, const int* fishId, float* phase, int N, float dt){
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(k >= N){
        return;
    }

    float ph = phase[fishId[k]] + (6.2831853f * length(vel[k]) * dt / (STRIDE * KOI_LEN));
    phase[fishId[k]] = (ph < 6.2831853f) ? ph : (ph - 6.2831853f);   // keep it small
}

// One thread per model point per koi: block x is the koi's slot, block y and
// thread x count the point. Move the point with its koi, find its cell, light
// it, and offer it to the cell.
__global__ void k_draw_fish(Screen sc, Camera cam, const KoiPoint* model, int nPts,
                            Patterns pat, unsigned seed, const float3* pos, const float3* vel, const int* fishId){
    int k = blockIdx.x;                                  // this block's koi, by its slot
    int i = (blockIdx.y * blockDim.x) + threadIdx.x;     // this thread's model point
    if(i >= nPts){
        return;
    }
    KoiPoint m  = model[i];
    int      id = fishId[k];                             // which koi is in slot k this frame
    float3   v  = vel[k];
    float    sp = length(v);
    float3   q  = m.p;                                   // the point, in the koi's own frame

    float ph = pat.phase[id] +(6.2831853f * hash01((unsigned)id)) - (WIG_K * m.u);
    q.z += (WIG_A + (WIG_B * sp)) * m.u * m.u * sinf(ph);
    // Turn: the koi's three axes from its velocity, then move it to its place.
    float3 f  = v * (1.0f / sp);                         // forward
    float3 r  = normalize(make_float3(-f.z, 0.0f, f.x)); // to its side, always level
    float3 up = cross(r, f);                             // up: at right angles to both
    float3 P  = pos[k] + (f * q.x) + (up * q.y) + (r * q.z);
    float3 N  = (f * m.n.x) + (up * m.n.y) + (r * m.n.z);

    int   cx, cy;                                        // the point's cell
    float z;                                             // and its depth
    if(!toCell(sc, cam, P, cx, cy, z)){
        return;
    }
    if(m.part == P_EYE){                                 // so that an eye wins its cell
        z -= 1.0f;
    }

    // Light: Lambert, plus a rim where the skin turns away from the camera.
    float nv = -dot(N, cam.fwd);                         // 1 facing the camera, 0 edge-on
    if(nv < 0.0f){                                       // it faces away: light its other side
        N  = N * -1.0f;
        nv = -nv;
    }
    float shade = 0.35f + (0.50f * fmaxf(dot(N, cam.light), 0.0f)) + (0.20f * powf(1.0f - nv, 3.0f));

    // The look. A highlight where the skin mirrors the light into the camera;
    // gold is metallic, so its highlight is three times as strong. Fins are
    // darker, with three rays across each. And the deeper a koi swims, the
    // darker it is.
    int    col  = koiColor(m, pat, id, seed);
    float3 half = normalize(cam.light - cam.fwd);        // halfway between the light and the camera
    shade += ((col == C_GOLD) ? 0.90f : 0.30f) * powf(fmaxf(dot(N, half), 0.0f), 14.0f);
    if(m.part != P_BODY && m.part != P_EYE){             // fins: darker, three rays across
        shade *= 0.70f - (0.15f * cosf(18.85f * m.v));
    }
    shade *= 1.0f - (DEEP * (1.0f - (P.y / TANK_Y)));
    // Offer the point to its cell. The smallest word wins: the nearest point.
    unsigned long long w = packCell(__float_as_uint(z), shade, m.part, col, id);
    atomicMin(&sc.depth[(cy * sc.cols) + cx], w);
}

__global__ void k_fade(Screen sc, float decay){
    int cx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int cy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(cx >= sc.cols || cy >= sc.rows){
        return;
    }
    int c = (cy * sc.cols) + cx;
    sc.color[c] = sc.color[c] * decay;

}

// Is cell (x, y) a gap, seen from a koi cell at depth z that belongs to koi
// `fish`? A gap is water, something much deeper, or another koi behind it.
__device__ inline int gapAt(Screen sc, int x, int y, float z, int fish){
    if(x < 0 || y < 0 || x >= sc.cols || y >= sc.rows){    // off the screen: no edge
        return 0;
    }
    unsigned long long w = sc.depth[(y * sc.cols) + x];
    if(w == EMPTY){
        return 1;
    }
    float nz = cellDepth(w);
    return (nz > (z + EDGE_Z)) || ((cellFish(w) != fish) && (nz > z));
}

__global__ void k_resolve_cells(Screen sc){
    int cx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int cy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(cx >= sc.cols || cy >= sc.rows){
        return;
    }

    int c = (cy * sc.cols) + cx;
    unsigned long long w = sc.depth[c];
    if(w == EMPTY){
        float3 old = sc.color[c];                        // what a koi left here, dimmed by k_fade
        float  top = fmaxf(old.x, fmaxf(old.y, old.z));
        sc.glyph[c] = (top > 0.35f) ? 3 : ((top > 0.06f) ? 1 : 0);   // ':', then '.', then nothing
        return;
    }
    float shade = cellShade(w), z = cellDepth(w);
    int   part = cellPart(w), col = cellColor(w), fish = cellFish(w);

    //The letter: from the ramp by brightness, or an letter where the koi
    // ends. gx and gyy say which side the gap is one: +1 right or below, -1 left
    // or above, 0 neigher or both. An eye is an o.

    int g = rampGlyph(shade);
    int gx = gapAt(sc, cx + 1, cy, z, fish) - gapAt(sc, cx - 1, cy, z, fish);
    int gy = gapAt(sc, cx, cy + 1, z, fish) - gapAt(sc, cx, cy - 1, z, fish);
    const int EDGE[9] = {0, 3, 2,  1, -1, 1,  2, 4, 0};  // rows: gap above, none, below
    int e = EDGE[((gy + 1) * 3) + gx + 1];
    bool edge = e >= 0;
    if(edge){
        g = G_EDGE + e;
    }
    if(part == P_EYE){
        g = G_BUBBLE;
    }
    sc.glyph[c] = (unsigned char)g;

    // The color: paint times brightness; brightness above 1 is highlight.
    float3 color = (koiPaint(col) * (0.35f + (0.65f * fminf(shade, 1.0f))))
                 + (make_float3(1.0f, 0.95f, 0.8f) * fmaxf(shade - 1.0f, 0.0f));
    if(edge){                                            // an edge letter has little ink
        color = color * 1.6f;
    }
    if(edge && col == C_BLACK){                          // black: a blue rim
        color += make_float3(0.05f, 0.12f, 0.25f);
    }
    sc.color[c] = color;
}