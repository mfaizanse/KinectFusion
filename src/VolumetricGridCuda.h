#pragma once

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Macros.h"
#include "Utils.h"

#define N_FIXED 640*480

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(Matrix3f* intrinsic_matrix,
               Matrix4f* current_pose,
               Matrix4f* base_pose_inv,
               float* depth_im,
               int im_height, int im_width,
               int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
               float voxel_size,
               float trunc_margin,
               float* voxel_grid_TSDF, float* voxel_grid_weight) {

    // process each 3 dim of the voxel grid

    int pt_grid_z = blockIdx.x;
    int pt_grid_y = threadIdx.x;

    Matrix4f cam2base = (*base_pose_inv) * (*current_pose);

    for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; pt_grid_x++) {

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};

        // First translate voxel center to the current frame camera coordinates' origin
        tmp_pt[0] = pt_base_x - cam2base(0, 3);
        tmp_pt[1] = pt_base_y - cam2base(1, 3);
        tmp_pt[2] = pt_base_z - cam2base(2, 3);

        // Rotate the voxel center using the inverse of the rotation matrix
        //The inverse of a rotation matrix is its transpose!

        float pt_cam_x = cam2base(0, 0) * tmp_pt[0] + cam2base(1, 0) * tmp_pt[1] +
                         cam2base(2, 0) * tmp_pt[2];
        float pt_cam_y = cam2base(0, 1) * tmp_pt[0] + cam2base(1, 1) * tmp_pt[1] +
                         cam2base(2, 1) * tmp_pt[2];
        float pt_cam_z = cam2base(0, 2) * tmp_pt[0] + cam2base(1, 2) * tmp_pt[1] +
                         cam2base(2, 2) * tmp_pt[2];

        //if depth value is invalid, then process the next voxel
        if (pt_cam_z <= 0)
            continue;

        //convert vexel center from camera space to image space using intrinsic parameters
        int pt_pix_x = roundf((*intrinsic_matrix)(0, 0) * (pt_cam_x / pt_cam_z) + (*intrinsic_matrix)(0, 2));
        int pt_pix_y = roundf((*intrinsic_matrix)(1, 1) * (pt_cam_y / pt_cam_z) + (*intrinsic_matrix)(1, 2));

        //chcek if the x and y coordinates fits in the image boundries, if not then process next voxel
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
            continue;

        // retrieve the depth value from the depth map for the corresponding voxel in the world space
        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

        // Voxels that has invalid depth values from depth map can be eliminated
        // Also the depth values greater than certain threshold should be also eliminated because
        // sensors of the camera is not reliable at this point. This value can be adjusted
        if (depth_val <= 0 || depth_val > 6)
            continue;

        // depth difference between camera space and depth map
        float diff = depth_val - pt_cam_z;

        // SDF truncation.(Eq.9)
        if (diff <= -trunc_margin)
            continue;

        // Integrate
        int volume_idx =
                pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;

        float dist = fmin(1.0f, diff / trunc_margin); //(Eq.9)
        float weight_old = voxel_grid_weight[volume_idx];
        // simply update the new weight by adding 1.
        float weight_new = weight_old + 1.0f;
        voxel_grid_weight[volume_idx] = weight_new;
        // update the voxel distances by taking the running average over all frames
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
    }
}


/**
 * VolumetricGridCuda - using CUDA
 */
class VolumetricGridCuda {
public:
    VolumetricGridCuda(Matrix3f* intrinsics_gpu, Matrix4f* base_pose_CPU) {
        //// ###  Initialize  Voumetric Grid
        voxel_grid_origin_x = -1.5f;
        voxel_grid_origin_y = -1.5f;
        voxel_grid_origin_z = 0.5f;

        // size of each voxel
        voxel_size = 0.01f;
        // define a margin for TSDF(can be adjusted depending on the problem)
        trunc_margin = voxel_size * 5;

        // size of the 3 dim voxel grid
        voxel_grid_dim_x = 500;
        voxel_grid_dim_y = 500;
        voxel_grid_dim_z = 500;

        voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
        voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
        for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {
            voxel_grid_TSDF[i] = 1.0f;
            voxel_grid_weight[i] = 0.0;
        }

        // allocate memory in the device for weights and distance of TSDF grid
        CUDA_CALL(cudaMalloc(&voxel_grid_TSDF_GPU, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));
        CUDA_CALL(cudaMalloc(&voxel_grid_weight_GPU, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));

        // copy distance and weights from host(CPU) to device(GPU) to previously allocated memory above
        CUDA_CALL(cudaMemcpy(voxel_grid_TSDF_GPU, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(voxel_grid_weight_GPU, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));

        //// ###  Initialize other params

        intrinsics = intrinsics_gpu;

        Matrix4f base_pose_inv_tmp = base_pose->inverse();

        CUDA_CALL(cudaMemcpy(base_pose, base_pose_CPU->data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(base_pose_inv, base_pose_inv_tmp.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));



    }
    ~VolumetricGridCuda() {
        CUDA_CALL(cudaFree(voxel_grid_TSDF_GPU));
        CUDA_CALL(cudaFree(voxel_grid_weight_GPU));
        CUDA_CALL(cudaFree(base_pose));
        CUDA_CALL(cudaFree(base_pose_inv));
        CUDA_CALL(cudaFree(intrinsics));

        free(voxel_grid_TSDF);
        free(voxel_grid_weight);
    }

    void integrateFrame(Matrix4f* current_pose_gpu, const FrameData& currentFrame) {
        std::cout << "Fusing into Volumetric Grid... " << std::endl;
        clock_t begin = clock();

        // currentFrame.depthMap  is already in GPU

        // Compute current frame camera pose relative to the base frame and store it in cam2base
//        multiply_matrix(base_pose_inv, current_pose, cam2base);

        // call integrate to to the integration in CUDA
        Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(
                intrinsics,
                current_pose_gpu,
                base_pose_inv,
                currentFrame.depthMap,
                currentFrame.height, currentFrame.width,
                voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                voxel_size,
                trunc_margin,
                voxel_grid_TSDF_GPU, voxel_grid_weight_GPU);

        CUDA_CHECK_ERROR

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Fusing Completed in " << elapsedSecs << " seconds." << std::endl;
    }

private:
    float voxel_grid_origin_x;
    float voxel_grid_origin_y;
    float voxel_grid_origin_z;
    // size of each voxel
    float voxel_size;
    // define a margin for TSDF(can be adjusted depending on the problem)
    float trunc_margin;
    // size of the 3 dim voxel grid
    int voxel_grid_dim_x;
    int voxel_grid_dim_y;
    int voxel_grid_dim_z;

    float* voxel_grid_TSDF;
    float* voxel_grid_weight;
    float* voxel_grid_TSDF_GPU;
    float* voxel_grid_weight_GPU;

    Matrix3f* intrinsics;
    Matrix4f* base_pose;
    Matrix4f* base_pose_inv;
};