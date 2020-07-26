//#pragma once
//
//#include "Utils.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "VolumetricGridCuda.h"
//
//__device__ __forceinline__
//float interpolate_trilinearly(const Vector3f& point,
//                              const float* volume,
//                              const int voxel_grid_dim_x,
//                              const int voxel_grid_dim_y,
//                              const int voxel_grid_dim_z){
//
//    Vector3i point_in_grid = point.cast<int>();
//
//    const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
//    const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
//    const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);
//
//    point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
//    point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
//    point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();
//
//    const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
//    const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
//    const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));
//
//    const int xd = point_in_grid.x();
//    const int yd = point_in_grid.y();
//    const int zd = point_in_grid.z();
//
//    const float c000 = volume[(xd) + (yd) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c001 = volume[(xd) + (yd) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c010 = volume[(xd) + (yd + 1) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c011 = volume[(xd) + (yd + 1) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c100 = volume[(xd + 1) + (yd) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c101 = volume[(xd + 1) + (yd) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c110 = volume[(xd + 1) + (yd + 1) * voxel_grid_dim_x + (zd) * voxel_grid_dim_x * voxel_grid_dim_y];
//    const float c111 = volume[(xd + 1) + (yd + 1) * voxel_grid_dim_x + (zd + 1) * voxel_grid_dim_x * voxel_grid_dim_y];
//
//
//    return c000 * (1 - a) * (1 - b) * (1 - c) +
//           c001 * (1 - a) * (1 - b) * c +
//           c010 * (1 - a) * b * (1 - c) +
//           c011 * (1 - a) * b * c +
//           c100 * a * (1 - b) * (1 - c) +
//           c101 * a * (1 - b) * c +
//           c110 * a * b * (1 - c) +
//           c111 * a * b * c;
//}
//
//__device__ __forceinline__
//float get_max_time(const Vector3f& volume_max, const Vector3f& origin, const Vector3f& direction){
//    float txmax = ((direction.x() > 0 ? volume_max.x() : 0.f) - origin.x()) / direction.x();
//    float tymax = ((direction.y() > 0 ? volume_max.y() : 0.f) - origin.y()) / direction.y();
//    float tzmax = ((direction.z() > 0 ? volume_max.z() : 0.f) - origin.z()) / direction.z();
//
//    return fmin(fmin(txmax, tymax), tzmax);
//}
//
//__device__ __forceinline__
//float get_min_time(const Vector3f& volume_max, const Vector3f& origin, const Vector3f& direction){
//    float txmin = ((direction.x() > 0 ? 0.f : volume_max.x()) - origin.x()) / direction.x();
//    float tymin = ((direction.y() > 0 ? 0.f : volume_max.y()) - origin.y()) / direction.y();
//    float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z()) - origin.z()) / direction.z();
//
//    return fmax(fmax(txmin, tymin), tzmin);
//}
//
//__global__
//void raycastTSDF(const float* voxel_grid_TSDF,
//                 const int voxel_grid_dim_x,
//                 const int voxel_grid_dim_y,
//                 const int voxel_grid_dim_z,
//                 const float voxel_size, // size of each voxel
//                 const float trunc_margin,
//                 const Matrix4f *pose,
//                 const Matrix3f *intrinsics,
//                 const size_t width,
//                 const size_t height,
//                 const size_t N,
//                 Vector3f* g_vertex,
//                 Vector3f* g_normal,
//                 float* renderedImage
//                 ) {
//
//    const bool INTERPOLATION_ENABLED = false;
//
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    //Terminate all un-necessary threads
//    if (idx >= N) {
//        return;
//    }
//
//    // Initialize the point and its normal
//    g_vertex[idx] = Vector3f(-MINF,-MINF,-MINF);
//    g_normal[idx] = Vector3f(-MINF,-MINF,-MINF);
//    renderedImage[idx] = 0;
//
//    // Get (x,y) of pixel
//    int x = idx / width;
//    int y = idx % width;
//
////    Matrix3f rotation = (*pose).block(0, 0, 3, 3);
////    Vector3f translation = (*pose).block(0, 3, 3, 1);
//
//    const Vector3f volume_range = Vector3f(voxel_grid_dim_x * voxel_size,
//                                           voxel_grid_dim_y * voxel_size,
//                                           voxel_grid_dim_z * voxel_size);
//
//    printf("21 \n");
//
//    Vector4f ray_start_tmp = ((*intrinsics).inverse() * Vector3f(x, y, 0)).homogeneous();
//    const auto rs1 = (*pose) * ray_start_tmp;
//
//    printf("22 \n");
//
//    Vector4f ray_next_tmp = ((*intrinsics).inverse() * Vector3f(x, y, 1.0f)).homogeneous();
//    const auto rs2 = (*pose) * ray_next_tmp;
//
//    printf("23 \n");
//
//    Vector3f ray_start = Vector3f(rs1.x(), rs1.y(),rs1.z());
//    Vector3f ray_next = Vector3f(rs2.x(), rs2.y(),rs2.z());
//    Vector3f ray_dir = (ray_next - ray_start).normalized();
//    float ray_len = 0;
//
//    printf("24 \n");
//
//
////    // @TODO: recheck it
////    const Vector3f pixel_position(
////            (x - (*intrinsics)(0,2)) / (*intrinsics)(0,0),
////            (y - (*intrinsics)(1,2)) / (*intrinsics)(1,1),
////            1.f);
////
////    Vector3f ray_direction = (rotation * pixel_position);
////    ray_direction.normalize();
////
////    float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
////    if(ray_length >= get_max_time(volume_range, translation, ray_direction)) {
////        return;
////    }
////
////    // @TODO: is voxel_scale == voxel_size????
////    ray_length += voxel_size;
//
//    ray_len += voxel_size/2;
//    Vector3f grid = (ray_start + (ray_dir * ray_len)) / voxel_size;
//
//    printf("25 \n");
//
//    // calculate index of grid in volume
//    int volume_idx1 =
//            (__float2int_rd(grid(2)) * voxel_grid_dim_y * voxel_grid_dim_x) + (__float2int_rd(grid(1)) * voxel_grid_dim_x) + __float2int_rd(grid(0));
//
//    printf("%f-%f-%f-%d\n", grid(2), grid(1), grid(0), volume_idx1);
//
//    float tsdf = voxel_grid_TSDF[volume_idx1];
////    if (INTERPOLATION_ENABLED) {
////        tsdf = interpolate_trilinearly(grid, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
////    }
////    else {
////        tsdf = voxel_grid_TSDF[volume_idx1];
////    }
//
//    //const float max_search_length = ray_length + volume_range.x() * sqrt(2.f);
//    const float voxelGridDiagonalLen = (Vector3f(voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z) - Vector3f(0,0,0)).norm();
//    const float max_search_length = voxelGridDiagonalLen * voxel_size * 2;
//    const float step_size = voxel_size;
//
//    printf("max: %f, step: %f", max_search_length, step_size);
//    bool was_ray_ever_inside = false;
//    //for(;ray_length < max_search_length; ray_length += trunc_margin * 0.5f){
//    for(;ray_len < max_search_length; ray_len += step_size){
//        printf("ray_len: %f", ray_len);
//        //grid = ((translation + (ray_direction * (ray_length + trunc_margin * 0.5f))) / voxel_size);
//
//        grid = (ray_start + (ray_dir * ray_len)) / voxel_size;
//
//        if (grid.x() < 1 || grid.x() >= voxel_grid_dim_x - 1 ||
//            grid.y() < 1 || grid.y() >= voxel_grid_dim_y -1 ||
//            grid.z() < 1 || grid.z() >= voxel_grid_dim_z - 1) {
////            if (was_ray_ever_inside) {
////                printf("breaking - was_ray_ever_inside \n");
////                break;
////            }
//            printf("continue - 11 \n");
//            continue;
//        }
//        was_ray_ever_inside = true;
//
//        volume_idx1 =
//                (__float2int_rd(grid(2)) * voxel_grid_dim_y * voxel_grid_dim_x) + (__float2int_rd(grid(1)) * voxel_grid_dim_x) + __float2int_rd(grid(0));
//
//        const float previous_tsdf = tsdf;
//        tsdf = voxel_grid_TSDF[volume_idx1];
//
//        if (previous_tsdf < 0.f && tsdf > 0.f) {
//            printf("breaking - Cross from -ve to +ve \n");
//            break;
//        }
//
//        if (previous_tsdf > 0.f && tsdf < 0.f) {
//            //const float t_star = ray_len - trunc_margin * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
//            // @TODO: check again this line
//            const float approx_len = ray_len - step_size * 0.5f * previous_tsdf / (tsdf - previous_tsdf);
//
//            const Vector3f vertex = ray_start + ray_dir * approx_len;
//
//            // const Vector3f location_in_grid = (vertex / voxel_size);
//            const Vector3f location_in_grid = vertex;
//            if (location_in_grid.x() < 1 || location_in_grid.x() >= voxel_grid_dim_x - 1 ||
//                location_in_grid.y() < 1 || location_in_grid.y() >= voxel_grid_dim_y - 1 ||
//                location_in_grid.z() < 1 || location_in_grid.z() >= voxel_grid_dim_z -1) {
//                printf("breaking for 0... \n");
//                break;
//            }
//
//            Vector3f normal, shifted;
//
//            shifted = location_in_grid;
//            shifted.x() += 1;
//            if (shifted.x() >= voxel_grid_dim_x - 1){
//                printf("breaking for 1...\n");
//                break;
//            }
//
//            const float Fx1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            shifted = location_in_grid;
//            shifted.x() -= 1;
//            if (shifted.x() < 1){
//                printf("breaking for 2... \n");
//                break;
//            }
//            const float Fx2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            normal.x() = (Fx1 - Fx2);
//
//            shifted = location_in_grid;
//            shifted.y() += 1;
//            if (shifted.y() >= voxel_grid_dim_y - 1){
//                printf("breaking for 3... \n");
//                break;
//            }
//            const float Fy1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            shifted = location_in_grid;
//            shifted.y() -= 1;
//            if (shifted.y() < 1){
//                printf("breaking for 4... \n");
//                break;
//            }
//            const float Fy2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            normal.y() = (Fy1 - Fy2);
//
//            shifted = location_in_grid;
//            shifted.z() += 1;
//            if (shifted.z() >= voxel_grid_dim_z - 1){
//                printf("breaking for 5... \n");
//                break;
//            }
//            const float Fz1 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            shifted = location_in_grid;
//            shifted.z() -= 1;
//            if (shifted.z() < 1){
//                printf("breaking for 6... \n");
//                break;
//            }
//            const float Fz2 = interpolate_trilinearly(shifted, voxel_grid_TSDF, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
//
//            normal.z() = (Fz1 - Fz2);
//
//            if (normal.norm() == 0){
//                printf("breaking for 7... \n");
//                break;
//            }
//
//            normal.normalize();
//
//            printf("FOUND CROSSING... \n");
//            g_vertex[idx] = Vector3f(vertex.x(), vertex.y(), vertex.z());
//            g_normal[idx] = Vector3f(normal.x(), normal.y(), normal.z());
//
//            //renderedImage[idx] = ray_direction.transpose() * g_normal[idx];
//            renderedImage[idx] = g_normal[idx].dot((Vector3f(1,1,1).normalized()));
//
//            break;
//        }
//    }
//
//    if (ray_len > max_search_length) {
//        printf("ray not hit anything...\n");
//    }
//
//}
//
//
//
//class SurfacePredictionCuda {
//public:
//    SurfacePredictionCuda(Matrix3f* intrinsics_gpu, cudaStream_t stream = 0) {
//        intrinsics = intrinsics_gpu;
//        this->stream = stream;
//
//        CUDA_CALL(cudaMalloc((void **) &pose_gpu, sizeof(Matrix4f)));
//    }
//
//    ~SurfacePredictionCuda() {
//        CUDA_CALL(cudaFree(pose_gpu));
//    }
//
//    void predict(const VolumetricGridCuda& volume,
//               Vector3f* g_vertices,
//               Vector3f* g_normals,
//               float* renderedImage,
//               const Matrix4f& pose,
//               size_t width,
//               size_t height
//               ) {
//
//        // copy pose to gpu
//        CUDA_CALL(cudaMemcpy(pose_gpu, pose.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
//
//        std::cout << "Surface Prediction ... " << std::endl;
//        clock_t begin = clock();
//
//        const size_t N = width * height;
//
//        raycastTSDF<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream>>>(
//                volume.voxel_grid_TSDF_GPU,
//                volume.voxel_grid_dim_x,
//                volume.voxel_grid_dim_y,
//                volume.voxel_grid_dim_z,
//                volume.voxel_size,
//                volume.trunc_margin,
//                pose_gpu,
//                intrinsics,
//                width,
//                height,
//                N,
//                g_vertices,
//                g_normals,
//                renderedImage
//                );
//
//        CUDA_CHECK_ERROR
//
//        // Wait for GPU to finish before accessing on host
//        cudaDeviceSynchronize();
//
//        clock_t end = clock();
//        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
//        // std::cout << "Surface Prediction Completed in " << elapsedSecs << " seconds." << std::endl;
//
//    }
//
//private:
//    Matrix3f* intrinsics;
//    cudaStream_t stream;
//    Matrix4f* pose_gpu;
//};