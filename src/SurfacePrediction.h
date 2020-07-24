#pragma once

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N_FIXED 640*480


__device__ __forceinline__
float interpolate_trilinearly(const Vector3f& point, const float* volume, const Vector3i& volume_size, const float voxel_scale){
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

    const float c000 = volume[x0 + y0 * (volume_size.y()-1) + z0 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c001 = volume[x1 + y0 * (volume_size.y()-1) + z0 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c010 = volume[x0 + y1 * (volume_size.y()-1) + z0 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c011 = volume[x1 + y1 * (volume_size.y()-1) + z0 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c100 = volume[x0 + y0 * (volume_size.y()-1) + z1 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c101 = volume[x1 + y0 * (volume_size.y()-1) + z1 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c110 = volume[x1 + y1 * (volume_size.y()-1) + z0 * (volume_size.z()-1) * (volume_size.y() -1)];
    const float c111 = volume[x1 + y1 * (volume_size.y()-1) + z1 * (volume_size.z()-1) * (volume_size.y() -1)];

    const float c00 = c000 * (1 - xd) + c100 * xd;
    const float c01 = c001 * (1 - xd) + c101 * xd;
    const float c10 = c010 * (1 - xd) + c110 * xd;
    const float c11 = c011 * (1 - xd) + c111 * xd;

    const float c0 = c00 * (1 - yd) + c10 * yd;
    const float c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;

    /*
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

    */
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
void raycast_tsdf_kernel(float* tsdf_volume,
                         Vector3f* g_vertex, Vector3f* g_normal,
                         const Vector3i volume_size, const float voxel_scale,
                         const Matrix3f cam_parameters,
                         const float truncation_distance,
                         const Matrix3f rotation,
                         const Vector3f translation) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= 640 || y >= 480)
        return;

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



class SurfacePrediction {
public:
    SurfacePrediction() {

    }
    ~SurfacePrediction() {

    }
    static void surface_prediction(const VolumetricGridCuda volume,
                                   Vector3f* g_vertices, Vector3f* g_normals,
                                   const Matrix3f cam_params,
                                   const Matrix4f pose) {

        dim3 threads(32, 32);
        dim3 blocks((640 + threads.x - 1) / threads.x,
                    (480 + threads.y - 1) / threads.y);

        raycast_tsdf_kernel<<<blocks, threads>>>(volume.voxel_grid_TSDF,
                                                 g_vertices, g_normals,
                                                 volume.voxel_grid_size, volume.voxel_size,
                                                 cam_params,
                                                 volume.trunc_margin,
                                                 pose.block(0, 0, 3, 3), pose.block(0, 3, 3, 1));
        cudaThreadSynchronize();

    }

private:
};