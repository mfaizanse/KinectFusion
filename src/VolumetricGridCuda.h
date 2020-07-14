#pragma once

#include "Utils.h"

#define N_FIXED 640*480

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * intrinsic_matrix, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

    // process each 3 dim of the voxel grid

    int pt_grid_z = blockIdx.x;
    int pt_grid_y = threadIdx.x;

    for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; pt_grid_x++) {

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};

        // First translate voxel center to the current frame camera coordinates' origin
        tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];

        // Rotate the voxel center using the inverse of the rotation matrix
        //The inverse of a rotation matrix is its transpose!

        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] +
                         cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] +
                         cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] +
                         cam2base[2 * 4 + 2] * tmp_pt[2];

        //if depth value is invalid, then process the next voxel
        if (pt_cam_z <= 0)
            continue;

        //convert vexel center from camera space to image space using intrinsic parameters
        int pt_pix_x = roundf(intrinsic_matrix[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + intrinsic_matrix[0 * 3 + 2]);
        int pt_pix_y = roundf(intrinsic_matrix[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + intrinsic_matrix[1 * 3 + 2]);

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
    VolumetricGridCuda() {}
    ~VolumetricGridCuda() {}

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f& initialPose) override {

        const size_t N = currentFrame.width * currentFrame.height;
        // The initial estimate can be given as an argument.
        Matrix4f *estimatedPose;
        CUDA_CALL(cudaMalloc((void **) &estimatedPose, sizeof(Matrix4f)));
        CUDA_CALL(cudaMemcpy(estimatedPose, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToDevice));

        Matrix4f *estimatedPose_cpu;
        estimatedPose_cpu = (Matrix4f*) malloc(sizeof(Matrix4f));
        CUDA_CALL(cudaMemcpy(estimatedPose_cpu, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToHost));

        std::cout << "Fusing into Volumetric Grid... " << std::endl;
        clock_t begin = clock();

        // call integrate to to the integration in CUDA
        Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(intrinsic_matrix_toGPU, cam2base_toGPU, depth_im_toGPU,
                im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                voxel_grid_TSDF_toGPU, voxel_grid_weight_toGPU);

        CUDA_CHECK_ERROR

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Fusing Completed in " << elapsedSecs << " seconds." << std::endl;

        CUDA_CALL(cudaFree(A));
        CUDA_CALL(cudaFree(b));
        CUDA_CALL(cudaFree(estimatedPose));

        free(A_cpu);
        free(b_cpu);
        // @TODO: See how can we free this pointer??  do  we need to?
        return *estimatedPose_cpu;
    }

private:
    bool test;
};