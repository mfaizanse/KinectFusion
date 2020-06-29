#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>


// Compute surface points from TSDF voxel grid and save points to point cloud file
void SaveVoxelGrid2SurfacePointCloud(const std::string &file_name, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                                     float voxel_size, float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                                     float * voxel_grid_TSDF, float * voxel_grid_weight,
                                     float tsdf_thresh, float weight_thresh) {

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

// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]) {
    mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
    mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
    mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
    mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

    mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
    mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
    mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
    mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

    mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
    mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
    mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
    mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

    mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
    mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
    mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
    mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N) {
    std::cout << filename << std::endl;
    std::vector<float> matrix;
    FILE *fp = fopen(filename.c_str(), "r");
    for (int i = 0; i < M * N; i++) {
        float tmp;
        int iret = fscanf(fp, "%f", &tmp);
        matrix.push_back(tmp);
    }
    fclose(fp);
    return matrix;
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth) {
    cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (depth_mat.empty()) {
        std::cout << "Error: depth image file not read!" << std::endl;
        cv::waitKey(0);
    }
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            depth[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
            if (depth[r * W + c] > 6.0f) // Only consider depth < 6m
                depth[r * W + c] = 0;
        }
}

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16]) {
    float inv[16], det;
    int i;
    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
             m[4]  * m[11] * m[14] +
             m[8]  * m[6]  * m[15] -
             m[8]  * m[7]  * m[14] -
             m[12] * m[6]  * m[11] +
             m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
              m[4]  * m[10] * m[13] +
              m[8]  * m[5] * m[14] -
              m[8]  * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
             m[1]  * m[11] * m[14] +
             m[9]  * m[2] * m[15] -
             m[9]  * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
             m[0]  * m[11] * m[13] +
             m[8]  * m[1] * m[15] -
             m[8]  * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
             m[0]  * m[7] * m[14] +
             m[4]  * m[2] * m[15] -
             m[4]  * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
              m[0]  * m[6] * m[13] +
              m[4]  * m[1] * m[14] -
              m[4]  * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

int main() {
    // directory where intrinsic matrix is located
    std::string intrinsic_path = "../data/camera-intrinsics.txt";

    // directory where  RGBD frames and camera poses located
    std::string data_path = "../data/rgbd-frames";

    // frame id's can be found in data_path
    int base_frame_idx = 150;
    int first_frame_idx = 150;
    float num_frames = 50;

    // intrinsic matrix K
    float intrinsic_matrix[3 * 3];

    // camera pose of the base frame
    float base_pose[4 * 4];

    // camera pose of the currently processed frame
    float current_pose[4 * 4];

    //camera pose of the currently processed frame RELATIVE TO THE BASE FRAME
    float cam2base[4 * 4];

    //boundries of the image space
    int im_width = 640;
    int im_height = 480;
    float depth_im[im_height * im_width];

    // Voxel grid parameters (change these to change voxel grid resolution, etc.)
    // Location of voxel grid origin in base frame camera coordinates
    float voxel_grid_origin_x = -1.5f;
    float voxel_grid_origin_y = -1.5f;
    float voxel_grid_origin_z = 0.5f;

    // size of each voxel
    float voxel_size = 0.01f;
    // define a margin for TSDF(can be adjusted depending on the problem)
    float trunc_margin = voxel_size * 5;

    // size of the 3 dim voxel grid
    int voxel_grid_dim_x = 500;
    int voxel_grid_dim_y = 500;
    int voxel_grid_dim_z = 500;

    // Read camera intrinsics and assign it to intrinsic matrix
    std::vector<float> K_temp = LoadMatrixFromFile(intrinsic_path, 3, 3);
    std::copy(K_temp.begin(), K_temp.end(), intrinsic_matrix);

    // Read base frame camera pose and assign it to base_pose
    std::ostringstream base_frame_prefix;
    base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
    std::string base_pose_file = data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
    std::vector<float> base_pose_temp = LoadMatrixFromFile(base_pose_file, 4, 4);
    std::copy(base_pose_temp.begin(), base_pose_temp.end(), base_pose);


    // Invert base frame camera pose to get extrinsic matrix
    float base_pose_inv[16] = {0};
    invert_matrix(base_pose, base_pose_inv);

    // Initialize voxel grid with 0 weights and 1 distance values for each voxel
    float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {
        voxel_grid_TSDF[i] = 1.0f;
        voxel_grid_weight[i] = 0.0;
    }

    // Each depth frame is processed and integrated into voxel grid by updating each voxel's distance and weight values
    for(int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; frame_idx++) {

        std::ostringstream curr_frame_prefix;
        curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

        // // Read current frame depth and store it in depth_im
        std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
        ReadDepth(depth_im_file, im_height, im_width, depth_im);

        // Read current frame camera pose
        std::string current_pose_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
        std::vector<float> current_pose_temp = LoadMatrixFromFile(current_pose_file, 4, 4);
        std::copy(current_pose_temp.begin(), current_pose_temp.end(), current_pose);

        // Compute current frame camera pose relative to the base frame and store it in cam2base
        multiply_matrix(base_pose_inv, current_pose, cam2base);

        std::cout << "Fusing: " << depth_im_file << std::endl;

        // process each 3 dim of the voxel grid
        for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; pt_grid_x++) {
            for (int pt_grid_y = 0; pt_grid_y < voxel_grid_dim_y; pt_grid_y++) {
                for (int pt_grid_z = 0; pt_grid_z < voxel_grid_dim_z; pt_grid_z++) {
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
        }

    }

    // Compute surface points from TSDF voxel grid and save to point cloud .ply file
    std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;

    SaveVoxelGrid2SurfacePointCloud("../data/tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                    voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                    voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

    return 0;
}
