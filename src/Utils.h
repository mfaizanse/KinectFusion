#pragma once

#include "Eigen.h"

struct FrameData {
    float* depthMap;
    Vector3f *g_vertices;
    Vector3f *g_normals;
    size_t width;
    size_t height;
    Matrix4f* globalCameraPose;
};