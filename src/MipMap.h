__global__ void createMipMap(float *depthMapLarge, float *depthMapSmall, size_t width, size_t height) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= height * width) {
        return;
    }

    size_t u = idx / width;
    size_t v = idx % width;

    size_t ul = u * 2;
    size_t vl = v * 2;

    size_t idxl = ul * width + vl;

    //Block averaging the values
    depthMapSmall[idx] = (depthMapLarge[idxl] + depthMapLarge[idxl + 1] + depthMapLarge[idxl + (width * 2)] + depthMapLarge[idxl + (width * 2) + 1]) / 4.0f;
}

struct DepthMipMap {
    float *depthMap;
    size_t width;
    size_t height;
};

class MipMapGen {
public:
    static void createDepthMipMap(float *depthMap, std::vector<DepthMipMap> mipmaps, size_t width, size_t height, cudaStream_t stream = 0) {
        size_t N = (width / 2) * (height / 2);

        createMipMap<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>>(mipmaps[0].depthMap, mipmaps[1].depthMap,width / 2, height / 2);

        N = (width / 4) * (height / 4);

        createMipMap<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>>(mipmaps[1].depthMap, mipmaps[2].depthMap,width / 4, height / 4);

    }
};
