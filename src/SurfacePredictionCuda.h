#pragma once

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "VolumetricGridCuda.h"

__device__ __forceinline__
float getValueOfVolume(const Vector3i& point,
                              const float* volume,
                              const int voxel_grid_dim_x,
                              const int voxel_grid_dim_y,
                              const int voxel_grid_dim_z){

    const int xd = point.x();
    const int yd = point.y();
    const int zd = point.z();

    return volume[(xd) + (yd) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
}

__device__ __forceinline__
float trilinearInterpolation(const Vector3f& point,
                              const float* volume,
                              const int voxel_grid_dim_x,
                              const int voxel_grid_dim_y,
                              const int voxel_grid_dim_z){

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

    const int xd = point_in_grid.x();
    const int yd = point_in_grid.y();
    const int zd = point_in_grid.z();

    const float c000 = volume[(xd) + (yd) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c001 = volume[(xd) + (yd) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c010 = volume[(xd) + (yd + 1) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c011 = volume[(xd) + (yd + 1) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c100 = volume[(xd + 1) + (yd) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c101 = volume[(xd + 1) + (yd) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c110 = volume[(xd + 1) + (yd + 1) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
    const float c111 = volume[(xd + 1) + (yd + 1) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];


    return c000 * (1 - a) * (1 - b) * (1 - c) +
           c001 * (1 - a) * (1 - b) * c +
           c010 * (1 - a) * b * (1 - c) +
           c011 * (1 - a) * b * c +
           c100 * a * (1 - b) * (1 - c) +
           c101 * a * (1 - b) * c +
           c110 * a * b * (1 - c) +
           c111 * a * b * c;
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
        return;
    }

    // Initialize the point and its normal
    g_vertex[idx] = Vector3f(-MINF,-MINF,-MINF);
    g_normal[idx] = Vector3f(-MINF,-MINF,-MINF);


    // Get (x,y) of pixel
//    int x = idx / width;
//    int y = idx % width;
    int x = idx / width;
    int y = height - (idx % width);

//    Matrix3f rotation = (*pose).block(0, 0, 3, 3);
//    Vector3f translation = (*pose).block(0, 3, 3, 1);

    //// backproject pixel (x, y ,0) and (x, y, 1) and trasnform
    Vector4f ray_start_tmp = ((*intrinsics).inverse() * Vector3f(x, y, 0)).homogeneous();
    const auto rs1 = (*pose) * ray_start_tmp;

    Vector4f ray_next_tmp = ((*intrinsics).inverse() * Vector3f(x, y, 1.0f)).homogeneous();
    const auto rs2 = (*pose) * ray_next_tmp;

    Vector3f ray_start = Vector3f(rs1.x(), rs1.y(),rs1.z());
    Vector3f ray_next = Vector3f(rs2.x(), rs2.y(),rs2.z());
    Vector3f ray_dir = (ray_next - ray_start).normalized();
    float ray_len = 0;

    // Translation vector to cater volume origin
    Vector3f vTranslate = Vector3f(-1.5f, -1.5f, 0.5f);

    // Now find the first voxel along ray_dir
    Vector3f grid = Vector3f(-MINF,-MINF,-MINF);
    const float voxelGridDiagonalLen = (Vector3f(voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z) - Vector3f(0,0,0)).norm();
    float min_search_length = 0.0;
    float max_search_length = voxelGridDiagonalLen * voxel_size;
    const float step_size = voxel_size/2;

    for (;ray_len < max_search_length; ray_len += step_size) {
        // calculate grid position
        grid = (ray_start + (ray_dir * ray_len) - vTranslate) / voxel_size;

        int x1 = __float2int_rd(grid(0));
        int y1 = __float2int_rd(grid(1));
        int z1 = __float2int_rd(grid(2));
        // If point outside of grid then continue
        if ((x1 < 1 || x1 >= voxel_grid_dim_x - 1)
            || (y1 < 1 || y1 >= voxel_grid_dim_y - 1)
            || (z1 < 1 || z1 >= voxel_grid_dim_z - 1)
        ) {
            // printf("1st point, continue: %d, %d, %d\n",  x1, y1, z1);
            continue;
        }

        // If point inside of grid then we found the first grid point
        min_search_length = ray_len;
        max_search_length = min_search_length + voxelGridDiagonalLen * voxel_size;
        break;
    }

    if (grid.x() == -MINF) {
        // No grid found
        printf("No grid found. return!");
        return;
    }

    // calculate index of grid in volume
    int volume_idx1 =
            (__float2int_rd(grid(2)) * voxel_grid_dim_y * voxel_grid_dim_x)
            + (__float2int_rd(grid(1)) * voxel_grid_dim_x)
            + __float2int_rd(grid(0));

    float tsdf = voxel_grid_TSDF[volume_idx1];

    //// Now run loop while ray is inside the grid and find the zero-crossing
    bool was_ray_ever_inside = false;
    for(;ray_len < max_search_length; ray_len += step_size){

        Vector3f grid = (ray_start + (ray_dir * ray_len) - vTranslate) / voxel_size;

        int x1 = __float2int_rd(grid(0));
        int y1 = __float2int_rd(grid(1));
        int z1 = __float2int_rd(grid(2));

        volume_idx1 = (z1 * voxel_grid_dim_y * voxel_grid_dim_x) + (y1 * voxel_grid_dim_x) + x1;

        if ((x1 < 1 || x1 >= voxel_grid_dim_x - 1)
            || (y1 < 1 || y1 >= voxel_grid_dim_y - 1)
            || (z1 < 1 || z1 >= voxel_grid_dim_z - 1)
        ) {
            if (was_ray_ever_inside) {
                break;
            }
            continue;
        }
        was_ray_ever_inside = true;

        const float previous_tsdf = tsdf;
        tsdf = voxel_grid_TSDF[volume_idx1];

        if (previous_tsdf < 0.f && tsdf > 0.f) {
            //printf("breaking - Cross from -ve to +ve \n");
            break;
        }

        //printf("PrevTSDF: %f, TSDF: %f\n", previous_tsdf, tsdf);
        if (previous_tsdf > 0.f && tsdf < 0.f) {
            const float approx_len = ray_len - step_size * 0.5f * previous_tsdf / (tsdf - previous_tsdf);

            // Compute 3d Point
            const Vector3f vertex = ray_start + (ray_dir * approx_len);

            const Vector3f location_in_grid = ((ray_start + (ray_dir * approx_len) - vTranslate) / voxel_size);
            if (location_in_grid.x() < 1
                || location_in_grid.x() >= voxel_grid_dim_x - 1
                || location_in_grid.y() < 1
                || location_in_grid.y() >= voxel_grid_dim_y - 1
                || location_in_grid.z() < 1
                || location_in_grid.z() >= voxel_grid_dim_z - 1) {

                break;
            }

            //// Compute normal
            Vector3f normal, shifted;

            shifted = location_in_grid;
            shifted.x() += 1;
            if (shifted.x() >= voxel_grid_dim_x - 1){
                break;
            }

            //const float Fx1 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fx1 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.x() -= 1;
            if (shifted.x() < 1){
                break;
            }
            //const float Fx2 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fx2 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.x() = (Fx1 - Fx2);

            shifted = location_in_grid;
            shifted.y() += 1;
            if (shifted.y() >= voxel_grid_dim_y - 1){
                break;
            }

            //const float Fy1 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fy1 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.y() -= 1;
            if (shifted.y() < 1){
                break;
            }
            //const float Fy2 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fy2 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.y() = (Fy1 - Fy2);

            shifted = location_in_grid;
            shifted.z() += 1;
            if (shifted.z() >= voxel_grid_dim_z - 1){
                break;
            }
            //const float Fz1 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fz1 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            shifted = location_in_grid;
            shifted.z() -= 1;
            if (shifted.z() < 1){
                break;
            }
            //const float Fz2 = getValueOfVolume(shifted.cast<int>(), voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
            const float Fz2 = trilinearInterpolation(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

            normal.z() = (Fz1 - Fz2);

            if (normal.norm() == 0){
                break;
            }

            normal.normalize();

            // Save results
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

        //std::cout << "Surface Prediction ... " << std::endl;

        const size_t N = width * height;

        //// For debugging, one thread at a time
//        for (int i=0;i< width; i++) {
//            for (int j=0;j< width; j++) {
//                raycastTSDF<<<1, 1, 0, stream>>>(
//                        volume.voxel_grid_TSDF_GPU,
//                        volume.voxel_grid_dim_x,
//                        volume.voxel_grid_dim_y,
//                        volume.voxel_grid_dim_z,
//                        volume.voxel_size,
//                        volume.trunc_margin,
//                        pose_gpu,
//                        intrinsics,
//                        width,
//                        height,
//                        N,
//                        g_vertices,
//                        g_normals,
//                        i*width+j
//                );
//
//                CUDA_CHECK_ERROR
//
//                // Wait for GPU to finish before accessing on host
//                cudaDeviceSynchronize();
//            }
//        }

        //// Running kernel for raycasting
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

    }

private:
    Matrix3f* intrinsics;
    cudaStream_t stream;
    Matrix4f* pose_gpu;
};