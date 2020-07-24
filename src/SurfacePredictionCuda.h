#pragma once

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VolumetricGridCuda.h"

//#define BLOCKSIZE 1024


__device__ __forceinline__
float interpolate_trilinearly(const Vector3f& point, const float* volume, const Vector3i& volume_size, const float voxel_scale){
    Vector3i point_in_grid = point.cast<int>();

    const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
    const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
    const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

    point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
    point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
    point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

    const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
    const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
    const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

    return static_cast<float>(volume[point_in_grid.x() * volume_size.x() + point_in_grid.y() * volume_size.y() + point_in_grid.z() * volume_size.z()]) * (1 - a) * (1 - b) * (1 - c) +
           static_cast<float>(volume[point_in_grid.x() * volume_size.x() + point_in_grid.y() * volume_size.y() + (point_in_grid.z() + 1) * volume_size.z()]) * (1 - a) * (1 - b) * c +
           static_cast<float>(volume[point_in_grid.x() * volume_size.x() + (point_in_grid.y() + 1) * volume_size.y() + point_in_grid.z() * volume_size.z()]) * (1 - a) * b * (1 - c) +
           static_cast<float>(volume[point_in_grid.x() * volume_size.x() + (point_in_grid.y() + 1) * volume_size.y() + (point_in_grid.z() + 1) * volume_size.z()]) * (1 - a) * b * c +
           static_cast<float>(volume[(point_in_grid.x() + 1) * volume_size.x() + point_in_grid.y() * volume_size.y() + point_in_grid.z() * volume_size.z()]) * a * (1 - b) * (1 - c) +
           static_cast<float>(volume[(point_in_grid.x() + 1) * volume_size.x() + point_in_grid.y() * volume_size.y() + (point_in_grid.z() + 1) * volume_size.z()]) * a * (1 - b) * c +
           static_cast<float>(volume[(point_in_grid.x() + 1) * volume_size.x() + (point_in_grid.y() + 1) * volume_size.y() + point_in_grid.z() * volume_size.z()]) * a * b * (1 - c) +
           static_cast<float>(volume[(point_in_grid.x() + 1) * volume_size.x() + point_in_grid.y() * volume_size.y() + (point_in_grid.z() + 1) * volume_size.z()]) * a * b * c;


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
                 const Matrix4f& pose,
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
        return;
    }

    int x = 2;
    int y = 3;

    const Vector3f volume_range = Vector3f(volume_size.x() * voxel_scale,
                                           volume_size.y() * voxel_scale,
                                           volume_size.z() * voxel_scale);

    const Vector3f pixel_position(
            (x - cam_parameters(0,2)) / cam_parameters(0,0),
            (y - cam_parameters(1,2)) / cam_parameters(1,1),
            1.f);

    Vector3f ray_direction = (rotation * pixel_position);
    ray_direction.normalize();

    float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
    if(ray_length >= get_max_time(volume_range, translation, ray_direction))
        return;

    ray_length += voxel_scale;
    Vector3f grid = (translation + (ray_direction * ray_length)) / voxel_scale;

    auto tsdf = static_cast<float>(tsdf_volume[__float2int_rd(grid(0)) + __float2int_rd(grid(1)) * volume_size.y() + __float2int_rd(grid(2)) * volume_size.z()]);

    const float max_search_length = ray_length + volume_range.x() * sqrt(2.f);
    for(;ray_length < max_search_length; ray_length += truncation_distance * 0.5f){
        grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);

        if (grid.x() < 1 || grid.x() >= volume_size.x() - 1 ||
            grid.y() < 1 || grid.y() >= volume_size.y() -1 ||
            grid.z() < 1 || grid.z() >= volume_size.z() - 1)
            continue;

        const float previous_tsdf = tsdf;
        tsdf = static_cast<float>(tsdf_volume[__float2int_rd(grid(0)) + __float2int_rd(grid(1)) * volume_size.y() + __float2int_rd(grid(2)) * volume_size.z()]);

        if (previous_tsdf < 0.f && tsdf > 0.f)
            break;
        if (previous_tsdf > 0.f && tsdf < 0.f) {
            const float t_star = ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);

            const auto vertex = translation + ray_direction * t_star;

            const Vector3f location_in_grid = (vertex / voxel_scale);
            if (location_in_grid.x() < 1 || location_in_grid.x() >= volume_size.x() - 1 ||
                location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y() - 1 ||
                location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z() -1)
                break;

            Vector3f normal, shifted;

            shifted = location_in_grid;
            shifted.x() += 1;
            if (shifted.x() >= volume_size.x() -1)
                break;
            const float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1)
                break;
            const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.x() = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= volume_size.y() -1)
                break;
            const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1)
                break;
            const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.y() = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= volume_size.z() -1)
                break;
            const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1)
                break;
            const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

            normal.z() = (Fz1 - Fz2);

            if (normal.norm() == 0)
                break;

            normal.normalize();
            g_vertex[y*640 + x] = Vector3f(vertex.x(), vertex.y(), vertex.z());
            g_normal[y*640 + x] = Vector3f(normal.x(), normal.y(), normal.z());

            break;
        }
    }

}



class SurfacePredictionCuda {
public:
    SurfacePredictionCuda(Matrix3f* intrinsics_gpu, cudaStream_t stream = 0) {
        intrinsics = intrinsics_gpu;
        this->stream = stream;
    }

    ~SurfacePredictionCuda() {

    }

    void predict(const VolumetricGridCuda& volume,
                                   Vector3f* g_vertices,
                                   Vector3f* g_normals,
                                   const Matrix4f& pose,
                                   size_t width,
                                   size_t height
                                   ) {

        const size_t N = width * height;

        raycastTSDF<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream>>>(
                volume.voxel_grid_TSDF_GPU,
                volume.voxel_grid_dim_x,
                volume.voxel_grid_dim_y,
                volume.voxel_grid_dim_z,
                volume.voxel_size,
                volume.trunc_margin,
                pose,
                intrinsics,
                width,
                height,
                N,
                g_vertices,
                g_normals
                );

        CUDA_CHECK_ERROR

        cudaThreadSynchronize();

    }

private:
    Matrix3f* intrinsics;
    cudaStream_t stream;
};