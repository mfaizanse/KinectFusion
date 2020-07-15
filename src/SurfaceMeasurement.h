#pragma once

#include "Eigen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Macros.h"
#include <cmath>

#define BLOCKSIZE 1024

/*
 * Compute vertices for valid depth measurements
 */
__global__ void measureSurfaceVertices(
        float *depthMap,
        Vector3f *vertices,
        Matrix3f *g_k_inv,
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

    //Back projection with filtered depth measurement
    if(depthMap[idx] <= 0.0f) {
        vertices[idx] = Vector3f(-MINF,-MINF,-MINF);
    } else {
        vertices[idx] = depthMap[idx] * g_k_inv[0] *
                        Vector3f(u, v, 1);
    }

}

/*
 * Compute normals for previously computed vertices
 */
__global__ void
measureSurfaceNormals(Vector3f *vertices, Vector3f *normals, size_t width, size_t height, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    Vector3f invalid = Vector3f(-MINF,-MINF,-MINF);

    if (u == 0 || v ==  0 || u == width-1 || v == height-1)  {
        normals[idx] = invalid;
    }
    else if(vertices[idx + width] != invalid && vertices[idx] != invalid && vertices[idx + 1] != invalid) {
        normals[idx] = (vertices[idx + width] - vertices[idx]).cross(vertices[idx + 1] - vertices[idx]).normalized();
    } else {
        normals[idx] = invalid;
    }
}

class SurfaceMeasurement {
public:
    /*
     * Set up surface measurement for specific camera and filtering parameters
     * k_inv represents camera intrinsics
     */
    SurfaceMeasurement(Matrix3f k_inv, float sigma_s, float sigma_r, cudaStream_t stream = 0) {

        CUDA_CALL(cudaMalloc((void **) &g_k_inv, sizeof(Matrix3f)));

        CUDA_CALL(cudaMemcpyAsync(g_k_inv, k_inv.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice, stream));

    }

    /*
     *  Compute both vertices and normals for depth mesurements
     *
     *  Works on different resolution for the pyramid of LOD's
     *
     *  Returns it's result in the g_vertices and g_normals vector, kept in GPU memory
     *  Validity map is not yet computed as
     */
    void measureSurface(size_t width, size_t height, Vector3f *g_vertices, Vector3f *g_normals, float *g_depthMap,
                        cudaStream_t stream = 0) {
        size_t sensorSize = width * height;

        measureSurfaceVertices<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>> (
                g_depthMap,
                g_vertices,
                g_k_inv,
                width,
                height,
                sensorSize
        );

        CUDA_CHECK_ERROR

        measureSurfaceNormals<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>> (
                g_vertices,
                g_normals,
                width,
                height,
                sensorSize
        );

        CUDA_CHECK_ERROR

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();
    }

    /*
     * Free GPU memory, only used in this class
     */
    ~SurfaceMeasurement() {
        CUDA_CALL(cudaFree(g_k_inv));

    }

private:
    Matrix3f *g_k_inv;
};

