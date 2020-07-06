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
};

/*
 * Inline this function for better readability
 */
__device__ __forceinline__ float fancyN(float sigma, float t) {
    pow(M_E, -pow(t, 2) / pow(sigma, 2));
}

/*
 * https://en.wikipedia.org/wiki/Bilateral_filter
 * Unsure which pixels need to be chosen for smoothing, maybe its only the surrounding 8 (maybe more?)
 * Wikipedia mentions the window centered in x
 * Sigma_r represents the range parameter, increases smoothing range
 * Sigma_s represents the spatial parameter, increases smoothing
 * Parameter Wp is now computed alongside, still not sure if this is correct
 */
__device__ float
computeDk(float *depthMap, size_t u, size_t v, float sigma_s, float sigma_r, size_t width, size_t N) {
    float sum = 0.0f;
    float sum2 = 0.0f;

    for (size_t i = 0; i < N; i++) {
        size_t u2 = i / width;
        size_t v2 = i % width;
        //Distance between pixels
        float t1 = sqrt(pow(u - u2, 2) + pow(v - v2, 2));
        //This can be implented far more efficient, but depends heavily on sigma_r
        if(t1 < 3 * sigma_r) {
            //Difference of depth measurements
            float t2 = abs(depthMap[u * width + v] - depthMap[i]);
            //Smoothing factor
            float fn_s = fancyN(sigma_s, t1);
            //Range factor
            float fn_r = fancyN(sigma_r, t2);
            //Weight
            float w = fn_s * fn_r;
            sum += w * depthMap[i];
            //Normalizing factor
            sum2 += w;
        }
    }

    //Return normalized result
    return sum / sum2;
}

/*
 * Compute vertices for valid depth measurements
 */
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

    //Back projection with filtered depth measurement
    vertices[idx] = computeDk(depthMap, u, v, consts.sigma_s, consts.sigma_r, width, N) * consts.g_k_inv[0] *
                    Vector3f(u, v, 1);

}

/*
 * Compute normals for previously computed vertices
 */
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

    /*
     * Set up surface mesurement for specific camera and filtering parameters
     * k_inv represents camera intrinsics
     */
    SurfaceMeasurement(Matrix3f k_inv, float sigma_s, float sigma_r, cudaStream_t stream = 0) {

        Matrix3f *g_k_inv;

        CUDA_CALL(cudaMalloc((void **) &g_k_inv, sizeof(Matrix3f)));

        CUDA_CALL(cudaMemcpyAsync(g_k_inv, k_inv.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice, stream));

        consts.g_k_inv = g_k_inv;
        consts.sigma_s = sigma_s;
        consts.sigma_r = sigma_r;

    }

    /*
     *  Compute both vertices and normals for depth mesurements
     *
     *  Works on different resolution for the pyramid of LOD's
     *
     *  Returns it's result in the g_vertices and g_normals vector, kept in GPU memory
     *  Validity map is not yet computed as
     */
    void measureSurface(size_t width, size_t height, Vector3f *g_vertices, Vector3f *g_normals, float *g_depthMap, bool *g_validityMap,
                        cudaStream_t stream = 0) {
        size_t sensorSize = width * height;

        measureSurfaceVertices<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (
                g_depthMap,
                g_vertices,
                consts,
                width,
                sensorSize
        );

        CUDA_CHECK_ERROR

        size_t normalsSize = (width - 1) * (height - 1);

        measureSurfaceNormals<<<(sensorSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >> > (
                g_vertices,
                g_normals,
                width - 1,
                height - 1,
                normalsSize
        );

        CUDA_CHECK_ERROR
    }

    /*
     * Free GPU memory, only used in this class
     */
    ~SurfaceMeasurement() {
        CUDA_CALL(cudaFree(consts.g_k_inv));
    }

private:
    constants consts;
};

