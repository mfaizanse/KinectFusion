#pragma once

#include "VirtualSensor.h"
#include "Eigen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Macros.h"

#define BLOCKSIZE 1024

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;
    // Color stored as 4 unsigned char
    Vector4uc color;
};

__global__ void backProjection(const float *depthMap, Vertex *vertices, Matrix4f *depthExtr, Matrix4f *camPosInv, float fovX, float fovY, float cX, float cY, size_t width, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if(idx >= N) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    float depth = depthMap[idx];

    vertices[idx].position = camPosInv[0] * depthExtr[0] * Vector4f((u - cX) / fovX * depth, (v - cY) / fovY * depth, depth, 1.0f);
}

class CudaMesh {

    CudaMesh(VirtualSensor& sensor, const Matrix4f& cameraPose, float edgeTreshhold = 0.01f) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        float* depthMap = sensor.getDepth();
        BYTE* colorMap = sensor.getColorRGBX();

        // Get depth intrinsics.
        Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();
        float fovX = depthIntrinsics(0, 0);
        float fovY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);

        // Compute inverse depth extrinsics.
        Matrix4f depthExtrinsicsInv = sensor.getDepthExtrinsics().inverse();

        // Compute inverse camera pose (mapping from camera CS to world CS).
        Matrix4f cameraPoseInverse = cameraPose.inverse();

        size_t sensorSize = sensor.getDepthImageWidth() * sensor.getDepthImageHeight();

        // Compute vertices with back-projection.
        m_vertices.resize(sensorSize);

        float *g_depthMap;
        CUDA_CALL(cudaMalloc((void **) &g_depthMap, sensorSize * sizeof(float)));

        Vertex *g_vertices;
        CUDA_CALL(cudaMalloc((void **) &g_vertices,sensorSize * sizeof(Vertex)));

        Matrix4f *g_camPos;
        CUDA_CALL(cudaMalloc((void **) &g_camPos, sizeof(Matrix4f)));

        Matrix4f *g_depthIntr;
        CUDA_CALL(cudaMalloc((void **) &g_depthIntr, sizeof(Matrix4f)));



        CUDA_CALL(cudaMemcpyAsync(g_depthMap,depthMap,sensorSize,cudaMemcpyHostToDevice, stream));

        CUDA_CALL(cudaMemcpyAsync(g_camPos,cameraPoseInverse.data(),sizeof(Matrix4f),cudaMemcpyHostToDevice,stream));

        CUDA_CALL(cudaMemcpyAsync(g_depthIntr,depthExtrinsicsInv.data(),sizeof(Matrix4f),cudaMemcpyHostToDevice,stream));

        backProjection<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream>>>(
                g_depthMap,
                g_vertices,
                g_camPos,
                g_depthIntr,
                fovX,fovY,
                cX,cY,
                sensor.getDepthImageWidth(),
                sensorSize
        );

        CUDA_CHECK_ERROR

        CUDA_CALL(cudaMemcpyAsync(m_vertices.data(),g_vertices,sensorSize,cudaMemcpyDeviceToHost,stream));

        CUDA_CALL(cudaDeviceSynchronize());
    }

private:
    std::vector<Vertex> m_vertices;
};
