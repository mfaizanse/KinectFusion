#pragma once

#include "VirtualSensor.h"
#include "Eigen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Macros.h"
#include <cmath>

#define BLOCKSIZE 1024

struct constants {
    Matrix3f *g_k_inv;
    float sigma_s;
    float sigma_r;
    float wp;
};

__device__ __forceinline__ float fancyN(float sigma, float t) {
    pow(M_E, -pow(t, 2) / pow(sigma, 2));
}

__device__ float
computeDk(float *depthMap, size_t u, size_t v, float sigma_s, float sigma_r, float wp, size_t width, size_t N) {
    float sum = 0.0f;

    for (size_t i = 0; i < N; i++) {
        size_t u2 = i / width;
        std::size_t v2 = i % width;
        float t1 = sqrt(pow(u - u2, 2) + pow(v - v2, 2));
        float t2 = abs(depthMap[u * width + v] - depthMap[i]);
        sum += fancyN(sigma_s, t1) * fancyN(sigma_r, t2) * depthMap[i];
    }

    return sum / wp;
}

__host__ __device__ void measureSurfaceVertices(
        float *depthMap,
        Vector3f *vertices,
        constants consts,
        size_t width,
        size_t height,
        size_t N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    vertices[idx] = computeDk(depthMap, u, v, consts.sigma_s, consts.sigma_r, consts.wp, width, N) * consts.g_k_inv[0] *
                    Vector3f(u, v, 1);

}

__host__ __device__ void
measureSurfaceNormals(Vector3f *vertices, Vector3f *normals, size_t width, size_t height, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    normals[idx] = (vertices[idx + width] - vertices[idx]).cross(vertices[idx + 1] - vertices[idx]).normalized();
}

class SurfaceMeasurement {

    SurfaceMeasurement(Matrix3f k_inv, float sigma_s, float sigma_r, float wp, cudaStream_t stream = 0) {

        Matrix3f *g_k_inv;

        CUDA_CALL(cudaMalloc((void **) &g_k_inv, sizeof(Matrix3f)));

        CUDA_CALL(cudaMemcpyAsync(g_k_inv, k_inv.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice, stream));

        consts.g_k_inv = g_k_inv;
        consts.sigma_s = sigma_s;
        consts.sigma_r = sigma_r;
        consts.wp = wp;

    }

    /*
     *  Sensor is only needed for sensor size, the rest should be pre-allocated, reusable GPU memory
     *
     *  Returns it's result in the g_vertices and g_normals vector, kept in GPU memory
     */
    void measureSurface(size_t width, size_t height, Vector3f *g_vertices, Vector3f *g_normals, float *g_depthMap,
                        cudaStream_t stream = 0) {
        size_t sensorSize = width * height;

        measureSurfaceVertices<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (
                g_depthMap,
                        g_vertices,
                        consts,
                        width,
                        sensorSize
        );

        size_t normalsSize = (width - 1) * (height - 1);

        measureSurfaceNormals<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (
                g_vertices,
                        g_normals,
                        width - 1,
                        height - 1,
                        normalsSize
        );
    }


    ~SurfaceMeasurement() {
        CUDA_CALL(cudaFree(consts.g_k_inv));
    }

private:
    constants consts;
};

