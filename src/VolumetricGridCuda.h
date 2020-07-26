#pragma once

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    Vector3i voxel_grid_size;

    float* voxel_grid_TSDF;
    float* voxel_grid_weight;
    float* voxel_grid_TSDF_GPU;
    float* voxel_grid_weight_GPU;

    Matrix3f* intrinsics;
    Matrix4f* base_pose;
    Matrix4f* base_pose_inv;

    VolumetricGridCuda(Matrix3f* intrinsics_gpu, Matrix4f* base_pose_CPU) {
        //global volume

        //// ###  Initialize  Voumetric Grid
//        voxel_grid_origin_x = 250.0f;
//        voxel_grid_origin_y = 250.0f;
//        voxel_grid_origin_z = 250.0f;

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
        voxel_grid_size = Vector3i(voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);

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

        Matrix4f base_pose_inv_tmp = base_pose_CPU->inverse();

        CUDA_CALL(cudaMalloc((void **) &base_pose, sizeof(Matrix4f)));
        CUDA_CALL(cudaMalloc((void **) &base_pose_inv, sizeof(Matrix4f)));

        CUDA_CALL(cudaMemcpy(base_pose, (*base_pose_CPU).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(base_pose_inv, base_pose_inv_tmp.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));



    }
    ~VolumetricGridCuda() {
        CUDA_CALL(cudaFree(voxel_grid_TSDF_GPU));
        CUDA_CALL(cudaFree(voxel_grid_weight_GPU));
        CUDA_CALL(cudaFree(base_pose));
        CUDA_CALL(cudaFree(base_pose_inv));

        free(voxel_grid_TSDF);
        free(voxel_grid_weight);
    }

    void integrateFrame(Matrix4f* current_pose, const FrameData& currentFrame) {
        // std::cout << "Fusing into Volumetric Grid... " << std::endl;
        clock_t begin = clock();

        Matrix4f* current_pose_gpu;
        CUDA_CALL(cudaMalloc((void **) &current_pose_gpu, sizeof(Matrix4f)));

        CUDA_CALL(cudaMemcpy(current_pose_gpu, (*current_pose).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

        // currentFrame.depthMap  is already in GPU

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

        CUDA_CALL(cudaFree(current_pose_gpu));

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        //std::cout << "Fusing Completed in " << elapsedSecs << " seconds." << std::endl;
    }

    // Compute surface points from TSDF voxel grid and save points to point cloud file
    void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, float tsdf_thresh, float weight_thresh) {

        // Compute surface points from TSDF voxel grid and save to point cloud .ply file
        std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;

        // Count total number of points in point cloud
        int num_pts = 0;
        for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
            if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
                num_pts++;

        // Create header for .ply file
        FILE *fp = fopen(file_name.c_str(), "w");
        fprintf(fp, "ply\n");
        fprintf(fp, "format binary_little_endian 1.0\n");
        fprintf(fp, "element vertex %d\n", num_pts);
        fprintf(fp, "property float x\n");
        fprintf(fp, "property float y\n");
        fprintf(fp, "property float z\n");
        fprintf(fp, "end_header\n");

        // Create point cloud content for ply file
        for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {
            // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
            if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh) {

                // Compute voxel indices in int for higher positive number range
                int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
                int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
                int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

                // Convert voxel indices to float, and save coordinates to ply file
                float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
                float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
                float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
                fwrite(&pt_base_x, sizeof(float), 1, fp);
                fwrite(&pt_base_y, sizeof(float), 1, fp);
                fwrite(&pt_base_z, sizeof(float), 1, fp);
            }
        }
        fclose(fp);
    }

    void SaveVoxelGrid(const std::string &voxel_grid_saveto_path) {
        // Save TSDF voxel grid and its parameters to disk as binary file (float array)
        std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;

        std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
        float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
        float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
        float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
        outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
        outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
        outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
        outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
        outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
        outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
        outFile.write((char*)&voxel_size, sizeof(float));
        outFile.write((char*)&trunc_margin, sizeof(float));
        for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
            outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
        outFile.close();
    }

    void copyVGFromDeviceToHost() {
        // Load TSDF voxel grid weights and distances from device back to host
        CUDA_CALL(cudaMemcpy(voxel_grid_TSDF, voxel_grid_TSDF_GPU,
                             voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float),
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(voxel_grid_weight, voxel_grid_weight_GPU,
                             voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
};