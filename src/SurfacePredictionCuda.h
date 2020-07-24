#pragma once

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VolumetricGridCuda.h"

__device__ __forceinline__
float interpolate_trilinearly(const Vector3f& point,
                              const float* volume,
                              const int voxel_grid_dim_x,
                              const int voxel_grid_dim_y,
                              const int voxel_grid_dim_z){

    Vector3i point_in_grid = point.cast<int>();

    const int x0 = round(point.x());
    const int y0 = round(point.y());
    const int z0 = round(point.y());

    const int x1 = (x0 > point.x()) ? x0 - 1 : x0 + 1;
    const int y1 = (y0 > point.y()) ? y0 - 1 : y0 + 1;
    const int z1 = (z0 > point.z()) ? z0 - 1 : z0 + 1;

    const float xd = (point.x() - x0) / (x1 - x0);
    const float yd = (point.y() - y0) / (y1 - y0);
    const float zd = (point.z() - z0) / (z1 - z0);

    // @TODO: Check the indexing, its different in our case
    const float c000 = volume[x0 + y0 * (voxel_grid_dim_y-1) + z0 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c001 = volume[x1 + y0 * (voxel_grid_dim_y-1) + z0 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c010 = volume[x0 + y1 * (voxel_grid_dim_y-1) + z0 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c011 = volume[x1 + y1 * (voxel_grid_dim_y-1) + z0 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c100 = volume[x0 + y0 * (voxel_grid_dim_y-1) + z1 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c101 = volume[x1 + y0 * (voxel_grid_dim_y-1) + z1 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c110 = volume[x1 + y1 * (voxel_grid_dim_y-1) + z0 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];
    const float c111 = volume[x1 + y1 * (voxel_grid_dim_y-1) + z1 * (voxel_grid_dim_z-1) * (voxel_grid_dim_y -1)];

    const float c00 = c000 * (1 - xd) + c100 * xd;
    const float c01 = c001 * (1 - xd) + c101 * xd;
    const float c10 = c010 * (1 - xd) + c110 * xd;
    const float c11 = c011 * (1 - xd) + c111 * xd;

    const float c0 = c00 * (1 - yd) + c10 * yd;
    const float c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}

__device__ __forceinline__
float get_max_time(const Vector3f& volume_max, const Vector3f& origin, const Vector3f& direction){
    float txmax = ((direction.x() > 0 ? volume_max.x() : 0.f) - origin.x()) / direction.x();
    float tymax = ((direction.y() > 0 ? volume_max.y() : 0.f) - origin.y()) / direction.y();
    float tzmax = ((direction.z() > 0 ? volume_max.z() : 0.f) - origin.z()) / direction.z();

    return fmin(fmin(txmax, tymax), tzmax);
}

__device__ __forceinline__
float get_min_time(const Vector3f& volume_max, const Vector3f& origin, const Vector3f& direction){
    float txmin = ((direction.x() > 0 ? 0.f : volume_max.x()) - origin.x()) / direction.x();
    float tymin = ((direction.y() > 0 ? 0.f : volume_max.y()) - origin.y()) / direction.y();
    float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z()) - origin.z()) / direction.z();

    return fmax(fmax(txmin, tymin), tzmin);
}

__global__
void raycastTSDF(const float* voxel_grid_TSDF,
                 const int voxel_grid_dim_x,
                 const int voxel_grid_dim_y,
                 const int voxel_grid_dim_z,
                 const float voxel_size, // size of each voxel
                 const float trunc_margin,
                 const Matrix4f *pose,
                 const Matrix3f *intrinsics,
                 const size_t width,
                 const size_t height,
                 const size_t N,
                 Vector3f* g_vertex,
                 Vector3f* g_normal
                 ) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        printf("a01-ending\n");
        return;
    }

    // Initialize the point and its normal
    g_vertex[idx] = Vector3f(-MINF,-MINF,-MINF);
    g_normal[idx] = Vector3f(-MINF,-MINF,-MINF);

    // Get (x,y) of pixel
    int x = idx / width;
    int y = idx % width;

    Matrix3f rotation = (*pose).block(0, 0, 3, 3);
    Vector3f translation = (*pose).block(0, 3, 3, 1);

    const Vector3f volume_range = Vector3f(voxel_grid_dim_x * voxel_size,
                                           voxel_grid_dim_y * voxel_size,
                                           voxel_grid_dim_z * voxel_size);


    // @TODO: recheck it
    const Vector3f pixel_position(
            (x - (*intrinsics)(0,2)) / (*intrinsics)(0,0),
            (y - (*intrinsics)(1,2)) / (*intrinsics)(1,1),
            1.f);

    Vector3f ray_direction = (rotation * pixel_position);
    ray_direction.normalize();

    float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
    if(ray_length >= get_max_time(volume_range, translation, ray_direction)) {
        return;
    }

    // @TODO: is voxel_scale == voxel_size????
    ray_length += voxel_size;

    Vector3f grid = (translation + (ray_direction * ray_length)) / voxel_size;

    // calculate index of grid in volume
    int volume_idx1 =
            (__float2int_rd(grid(2)) * voxel_grid_dim_y * voxel_grid_dim_x) + (__float2int_rd(grid(1)) * voxel_grid_dim_x) + __float2int_rd(grid(0));

    // printf("%f-%f-%f-%d\n", grid(2), grid(1), grid(0), volume_idx1);

    float tsdf = voxel_grid_TSDF[volume_idx1];

    const float max_search_length = ray_length + volume_range.x() * sqrt(2.f);
    for(;ray_length < max_search_length; ray_length += trunc_margin * 0.5f){
        grid = ((translation + (ray_direction * (ray_length + trunc_margin * 0.5f))) / voxel_size);

        if (grid.x() < 1 || grid.x() >= voxel_grid_dim_x - 1 ||
            grid.y() < 1 || grid.y() >= voxel_grid_dim_y -1 ||
            grid.z() < 1 || grid.z() >= voxel_grid_dim_z - 1)
            continue;

        volume_idx1 =
                (__float2int_rd(grid(2)) * voxel_grid_dim_y * voxel_grid_dim_x) + (__float2int_rd(grid(1)) * voxel_grid_dim_x) + __float2int_rd(grid(0));

        const float previous_tsdf = tsdf;
        tsdf = voxel_grid_TSDF[volume_idx1];

        if (previous_tsdf < 0.f && tsdf > 0.f)
            break;
        if (previous_tsdf > 0.f && tsdf < 0.f) {
            const float t_star = ray_length - trunc_margin * 0.5f * previous_tsdf / (tsdf - previous_tsdf);

            const auto vertex = translation + ray_direction * t_star;

            const Vector3f location_in_grid = (vertex / voxel_size);
            if (location_in_grid.x() < 1 || location_in_grid.x() >= voxel_grid_dim_x - 1 ||
                location_in_grid.y() < 1 || location_in_grid.y() >= voxel_grid_dim_y - 1 ||
                location_in_grid.z() < 1 || location_in_grid.z() >= voxel_grid_dim_z -1) {
                break;
            }

            printf("FOUND CROSSING... \n");

            Vector3f normal, shifted;

            shifted = location_in_grid;
            shifted.x() += 1;
            if (shifted.x() >= voxel_grid_dim_x - 1)
                break;
            const float Fx1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1)
                break;
            const float Fx2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.x() = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= voxel_grid_dim_y - 1)
                break;
            const float Fy1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1)
                break;
            const float Fy2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.y() = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= voxel_grid_dim_z - 1)
                break;
            const float Fz1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1)
                break;
            const float Fz2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.z() = (Fz1 - Fz2);

            if (normal.norm() == 0)
                break;

            normal.normalize();
            g_vertex[idx] = Vector3f(vertex.x(), vertex.y(), vertex.z());
            g_normal[idx] = Vector3f(normal.x(), normal.y(), normal.z());

            break;
        }
    }

}



class SurfacePredictionCuda {
public:
    SurfacePredictionCuda(Matrix3f* intrinsics_gpu, cudaStream_t stream = 0) {
        intrinsics = intrinsics_gpu;
        this->stream = stream;

        CUDA_CALL(cudaMalloc((void **) &pose_gpu, sizeof(Matrix4f)));
    }

    ~SurfacePredictionCuda() {
        CUDA_CALL(cudaFree(pose_gpu));
    }

    void predict(const VolumetricGridCuda& volume,
               Vector3f* g_vertices,
               Vector3f* g_normals,
               const Matrix4f& pose,
               size_t width,
               size_t height
               ) {

        // copy pose to gpu
        CUDA_CALL(cudaMemcpy(pose_gpu, pose.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

        std::cout << "Surface Prediction ... " << std::endl;
        clock_t begin = clock();

        const size_t N = width * height;

        raycastTSDF<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream>>>(
                volume.voxel_grid_TSDF_GPU,
                volume.voxel_grid_dim_x,
                volume.voxel_grid_dim_y,
                volume.voxel_grid_dim_z,
                volume.voxel_size,
                volume.trunc_margin,
                pose_gpu,
                intrinsics,
                width,
                height,
                N,
                g_vertices,
                g_normals
                );

        CUDA_CHECK_ERROR

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Surface Prediction Completed in " << elapsedSecs << " seconds." << std::endl;

    }

private:
    Matrix3f* intrinsics;
    cudaStream_t stream;
    Matrix4f* pose_gpu;
};