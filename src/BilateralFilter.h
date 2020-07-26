#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <limits>

#define BLOCKSIZE 1024

#define SKIP_FILTER 0

/*
 * Inline this function for better readability
 */
__device__ __forceinline__ float fancyN(float sigma, float t) {
    return pow(M_E, -pow(t, 2) / pow(sigma, 2));
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
computeDk(float *depthMap, long u, long v, float sigma_s, float sigma_r, size_t width, size_t N) {
    float sum = 0.0f;
    float sum2 = 0.0f;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //Use a 3x3 grid as the smoothing kernel
    for (size_t i = 0; i < 49; i++) {
        long u2 = u + (i/7) - 3;
        long v2 = v + (i%7) - 3;
        //Skip depth measurements over the edges of the image
        if(u2 < 0 || v2 < 0 || u2 == width || v2 == (N/width)){
            continue;
        }
        //Skip invalid depth measurements
        if(depthMap[u2 * width + v2] <= 0.0f) {
            continue;
        }
        //Distance between pixels
        float t1 = sqrt(pow(static_cast<float>(u2 - u), 2) + pow(static_cast<float>(v2 - v), 2));
        //Difference of depth measurements
        float t2 = abs(depthMap[u2 * width + v2] - depthMap[idx]);
        //Smoothing factor
        float fn_s = fancyN(sigma_s, t1);
        //Range factor
        float fn_r = fancyN(sigma_r, t2);
        //Weight
        float w = fn_s * fn_r;
        sum += w * depthMap[u2 * width + v2];
        //Normalizing factor
        sum2 += w;
    }
    //Return normalized result
    return sum / sum2;
}

__global__ void bilateralFilter(float *depthMap, float *filteredMap, float sigma_s, float sigma_r, size_t width, size_t height, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    if(SKIP_FILTER) {
        filteredMap[idx] = depthMap[idx];
        return;
    }

    if(depthMap[idx] <= 0) {
        filteredMap[idx] = depthMap[idx];
    } else {
        float f = computeDk(depthMap,u,v,sigma_s,sigma_r,width,N);
        filteredMap[idx] = f;
    }
}

class BilateralFilter {
public:
    static void filterDepthmap(float *depthMap, float *filteredMap, size_t width, float sigma_s, float sigmna_r, size_t height, size_t N, cudaStream_t stream = 0) {
        bilateralFilter<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>>(depthMap,filteredMap, sigma_s, sigmna_r, width,height,N);
    }
};
