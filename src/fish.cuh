#pragma once
#include "screen.cuh"
#include "model.cuh"

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
    float sx = sc.W * 0.5f + cam.zoom * dot(d, cam.right);   // pixel x
    float sy = sc.H * 0.5f - cam.zoom * dot(d, cam.up);      // pixel y, growing downward
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

// Temp

__global__ void k_gather_cells(Screen sc, Camera cam, const KoiPoint* model, int nPts){
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;

    if(cx >= sc.cols || cy >= sc.rows){
        return;
    }

    float best = 1e30f, shade = 0.0f;                    // the nearest depth so far, its brightness
    for(int i = 0; i < nPts; i++){
        KoiPoint m = model[i];
        float3 P = m.p;                                  // the koi sits at the origin
        int    px, py;                                   // the point's cell
        float  z;                                        // and its depth
        if(!toCell(sc, cam, P, px, py, z) || px != cx || py != cy){
            continue;
        }

        float3 N = m.n;
        if(dot(N, cam.fwd) > 0.0f){                      // it faces away: light its other side
            N = N * -1.0f;
        }
        if(z < best){                                    // nearer than what the cell has so far
            best  = z;
            shade = 0.35f + 0.50f * fmaxf(dot(N, cam.light), 0.0f);
        }
    }

    int c = cy * sc.cols + cx;
    sc.glyph[c] = (unsigned char)(best < 1e30f ? rampGlyph(shade) : 0);
    sc.color[c] = make_float3(0.92f, 0.90f, 0.84f) * shade;
}