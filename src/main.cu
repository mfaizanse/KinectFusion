#include <iostream>
#include <fstream>
#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "SurfaceMeasurement.h"
#include "SurfacePredictionCuda.h"
#include "CudaICPOptimizer.h"
#include "BilateralFilter.h"
#include "VolumetricGridCuda.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "VolumetricGridCuda.h"


#define USE_GPU_ICP	1
#define USE_REDUCTION_ICP 1
#define SHOW_MESH 0


int reconstructRoom() {
    // Setup virtual sensor
    std::string filenameIn = std::string("../../data/rgbd_dataset_freiburg1_xyz/");
    std::string filenameBaseOut = std::string("../../outputs/mesh_");


    // Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

    const unsigned depthFrameWidth = sensor.getDepthImageWidth();
    const unsigned depthFrameHeight = sensor.getDepthImageHeight();
    const size_t N = depthFrameWidth * depthFrameHeight;

    // Setup the ICP optimizer.
    ICPOptimizer* optimizer;
    if(USE_REDUCTION_ICP) {
        optimizer = new LinearICPCubOptimizer(depthFrameWidth,depthFrameHeight);
    } else {
        optimizer = new LinearICPCudaOptimizer(depthFrameWidth,depthFrameHeight);
    }

    optimizer->setMatchingMaxDistance(0.1f);
    //optimizer->setMatchingMaxDistance(0.0003f);
    optimizer->setMatchingMaxAngle(1.0472f); // 0.523599 // 1.0472f
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(20);

    // Intrinsics on host memory
    Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();

    // Intrinsics on device memory
    Matrix3f *cudaDepthIntrinsics;
    CUDA_CALL(cudaMalloc((void **) &cudaDepthIntrinsics, sizeof(Matrix3f)));
    CUDA_CALL(cudaMemcpy(cudaDepthIntrinsics, depthIntrinsics.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice));


    Matrix4f currentCameraToWorld = Matrix4f::Identity();
    Matrix4f base_pose_cpu = Matrix4f::Identity();

	// We store the estimated camera poses. [on Host memory]
	// estimated poses will save world to camera pose
	std::vector<Matrix4f> estimatedPoses;

    TransformHelper transformHelper;

    SurfaceMeasurement surfaceMeasurement(depthIntrinsics.inverse(), 0);
    VolumetricGridCuda volumetricGrid(cudaDepthIntrinsics,  &base_pose_cpu);
    SurfacePredictionCuda surfacePrediction(cudaDepthIntrinsics, 0);

    // Defining memory for previous and current frames,  [on Device memory]
    FrameData previousFrame;
    FrameData currentFrame;

    float *unfilteredDepth;

    previousFrame.width =  depthFrameWidth;
    previousFrame.height = depthFrameHeight;

    currentFrame.width =  depthFrameWidth;
    currentFrame.height = depthFrameHeight;

    CUDA_CALL(cudaMalloc((void **) &unfilteredDepth, N * sizeof(float)));

    CUDA_CALL(cudaMalloc((void **) &previousFrame.depthMap, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.renderedImage, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.g_vertices, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.g_normals, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.globalCameraPose, sizeof(Matrix4f)));

    CUDA_CALL(cudaMalloc((void **) &currentFrame.depthMap, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.renderedImage, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.g_vertices, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.g_normals, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.globalCameraPose, sizeof(Matrix4f)));

    CUDA_CALL(cudaMemcpy(previousFrame.globalCameraPose, currentCameraToWorld.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(currentFrame.globalCameraPose, currentCameraToWorld.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

    Matrix4f *cuda4fIdentity;
    CUDA_CALL(cudaMalloc((void **) &cuda4fIdentity, sizeof(Matrix4f)));
    CUDA_CALL(cudaMemcpy(cuda4fIdentity, currentCameraToWorld.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

    Matrix4f *tmp4fMat_cpu;
    tmp4fMat_cpu = (Matrix4f*) malloc(sizeof(Matrix4f));


    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Vector3f *g_vertices_host;
    g_vertices_host = (Vector3f *) malloc(N * sizeof(Vector3f));
  
    cv::Mat renderedDepthImg = cv::Mat::zeros(depthFrameHeight, depthFrameWidth, CV_32FC1);


	int i = 0;
	const int iMax = 5;
	while (sensor.processNextFrame() && i < iMax) {
	    // Get current depth frame
		float* depthMap = sensor.getDepth();

		// Copy depth map to current frame, device memory
        CUDA_CALL(cudaMemcpy(unfilteredDepth, depthMap, N * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(start);
        BilateralFilter::filterDepthmap(unfilteredDepth,currentFrame.depthMap,depthFrameWidth,100,3,depthFrameHeight,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        printf("Filter depthmap elapsed time : %f ms\n" ,elapsedTime);


        // #### Step 1: Surface measurement
        // It expects the pointers for device memory

        cudaEventRecord(start);
        surfaceMeasurement.measureSurface(depthFrameWidth, depthFrameHeight,
                                            currentFrame.g_vertices, currentFrame.g_normals, currentFrame.depthMap,
                                          0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        printf("Surface measurement elapsed time : %f ms\n" ,elapsedTime);

        ///// Debugging code start
        //// We write out the mesh to file for debugging.
//        Vector3f *g_vertices_host;
//        g_vertices_host = (Vector3f *) malloc(N * sizeof(Vector3f));
//        std::cout << "step 6" << std::endl;
//        CUDA_CALL(cudaMemcpy(g_vertices_host, currentFrame.g_vertices, N * sizeof(Vector3f), cudaMemcpyDeviceToHost));

//        SimpleMesh currentSM{ currentFrame.g_vertices, depthFrameWidth,depthFrameHeight, sensor.getColorRGBX(), 0.1f };
//        std::stringstream ss1;
//        ss1 << filenameBaseOut << "SM_" << sensor.getCurrentFrameCnt() << ".off";
//        if (!currentSM.writeMesh(ss1.str())) {
//            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
//            return -1;
//        }
//        free(g_vertices_host);
        ///// Debugging code  end

		// #### Step 2: Pose Estimation (Using Linearized ICP)
		Matrix4f currentFrameToPreviousFrame = Matrix4f::Identity();
		// Don't do ICP on 1st  frame
		if (i > 0) {
            if (USE_GPU_ICP)  {
                // The arguments should be on device memory
                // The returned pose matrix will be on host memory
                cudaEventRecord(start);
                currentFrameToPreviousFrame = optimizer->estimatePose(*cudaDepthIntrinsics, currentFrame, previousFrame, *previousFrame.globalCameraPose);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&elapsedTime, start,stop);
                printf("ICP elapsed time : %f ms\n" ,elapsedTime);
            }
            else {
                // currentCameraToWorld = optimizer->estimatePose(depthIntrinsics, currentFrame, previousFrame, Matrix4f::Identity());
            }
		}
        //std::cout << "currentFrameToPreviousFrame pose: " << std::endl << currentFrameToPreviousFrame << std::endl;
        currentCameraToWorld = currentFrameToPreviousFrame * currentCameraToWorld;

		//// Step 3:  Volumetric Grid Fusion

		// @TODO: copy  currentCameraToWorld  to gpu
		cudaEventRecord(start);

		volumetricGrid.integrateFrame(&currentCameraToWorld,  currentFrame);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        printf("Volumetric fusion elapsed time : %f ms\n" ,elapsedTime);


        // Step 4: Ray-Casting
        surfacePrediction.predict(volumetricGrid,
		        currentFrame.g_vertices,
		        currentFrame.g_normals,
		        currentFrame.renderedImage,
		        currentCameraToWorld,
                depthFrameWidth,
                depthFrameHeight);


		// Step 5: Update trajectory poses
		// Invert the transformation matrix to get the current camera pose.  [Host memory]
        Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		// Update globalCameraPose on device memory
        CUDA_CALL(cudaMemcpy(currentFrame.globalCameraPose, currentCameraPose.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

        // Render the raycast result
        CUDA_CALL(cudaMemcpy(renderedDepthImg.data, currentFrame.renderedImage, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(g_vertices_host, currentFrame.g_vertices, N * sizeof(Vector3f), cudaMemcpyDeviceToHost));
        std::cout << "Saving rendered mesh ..." << std::endl;
        SimpleMesh currentSM{ g_vertices_host, depthFrameWidth,depthFrameHeight, sensor.getColorRGBX(), false,0.1f };
        std::stringstream ss1;
        ss1 << filenameBaseOut << "SM_" << sensor.getCurrentFrameCnt() << ".off";
        if (!currentSM.writeMesh(ss1.str())) {
            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
            return -1;
        }

        // Step 7: Update data (e.g. Poses, depth frame etc.) for next frame
		// Update previous frame data
        FrameData tmpFrame = previousFrame;
        previousFrame = currentFrame;
        currentFrame = tmpFrame;

        if(SHOW_MESH) {
            cv::Mat img = cv::Mat::zeros(480, 640, CV_32F);

            std::vector<Vector3f> normals = std::vector<Vector3f>(640 * 480);

            CUDA_CALL(cudaMemcpyAsync(normals.data(), previousFrame.g_normals, sizeof(Vector3f) * 640 * 480,
                                      cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());

            std::cout << "Generating img" << std::endl;
            for (int normal_idx = 0; normal_idx < normals.size(); normal_idx++) {
                img.at<float>(normal_idx) = normals[normal_idx].dot((Vector3f(1, 1, 1).normalized()));
            }
            std::cout << "Done." << std::endl;

            cv::namedWindow("Current mesh");
            cv::imshow("Current mesh", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

		// if (i % 5 == 0) {

		if (i > 0) {
		    std::cout << "Saving mesh ..." << std::endl;
            // We write out the mesh to file for debugging.
            std::vector<Vector3f> cpu_vertices = std::vector<Vector3f>(640 * 480);
            CUDA_CALL(cudaMemcpyAsync(cpu_vertices.data(),previousFrame.g_vertices,sizeof(Vector3f) * 640 * 480,cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaDeviceSynchronize());
            SimpleMesh filteredDepthMesh{cpu_vertices.data(),640,480,sensor.getColorRGBX(), 0.1f};
            //SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
            SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
            SimpleMesh resultingMesh = SimpleMesh::joinMeshes(filteredDepthMesh, currentCameraMesh, Matrix4f::Identity());

            std::stringstream ss;
            ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
            if (!resultingMesh.writeMesh(ss.str())) {
                std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                return -1;
            }
		}

		i++;
	}

	//  Save Volumetric Grid as pointclouds
    std::stringstream ss2, ss3;
    ss2 << filenameBaseOut << "tsdf.ply";
    ss3 << filenameBaseOut << "tsdf.bin";
    volumetricGrid.copyVGFromDeviceToHost();
	volumetricGrid.SaveVoxelGrid2SurfacePointCloud(ss2.str(),  0.2f, 0.0f);
    volumetricGrid.SaveVoxelGrid(ss3.str());

	// Free all pointers
    CUDA_CALL(cudaFree(cudaDepthIntrinsics));

    CUDA_CALL(cudaFree(previousFrame.depthMap));
    CUDA_CALL(cudaFree(previousFrame.g_vertices));
    CUDA_CALL(cudaFree(previousFrame.g_normals));
    CUDA_CALL(cudaFree(previousFrame.globalCameraPose));

    CUDA_CALL(cudaFree(currentFrame.depthMap));
    CUDA_CALL(cudaFree(currentFrame.g_vertices));
    CUDA_CALL(cudaFree(currentFrame.g_normals));
    CUDA_CALL(cudaFree(currentFrame.globalCameraPose));

    free(tmp4fMat_cpu);
    free(g_vertices_host);

    delete optimizer;

	return 0;
}

int main() {
    int result = reconstructRoom();
	return result;
}
