#pragma once

#include "Eigen.h"

struct FrameData {
    float* depthMap; // On device memory
    Vector3f *g_vertices; // On device memory
    Vector3f *g_normals;  // On device memory
    size_t width;
    size_t height;
    Matrix4f* globalCameraPose; // On device memory
    float* renderedImage; // On device memory
};