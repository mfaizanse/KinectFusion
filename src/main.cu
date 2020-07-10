#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "SurfaceMeasurement.h"
#include "CudaICPOptimizer.h"

#define USE_GPU_ICP	1

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

	// sensor.processNextFrame();

    // Setup the ICP optimizer.
    ICPOptimizer* optimizer;

	if (USE_GPU_ICP)  {
	    optimizer = new LinearICPCudaOptimizer();
	}
	else {
        std::cout << "Using CPU ICP" << std::endl;
	    optimizer = new LinearICPOptimizer();
	}

    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->setMatchingMaxAngle(1.0472f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(10);

    const unsigned depthFrameWidth = sensor.getDepthImageWidth();
    const unsigned depthFrameHeight = sensor.getDepthImageHeight();
    const size_t N = depthFrameWidth * depthFrameHeight;

    // Intrinsics on host memory
    Matrix3f depthIntrinsics = sensor.getDepthIntrinsics();

    // Intrinsics on device memory
    Matrix3f *cudaDepthIntrinsics;
    CUDA_CALL(cudaMalloc((void **) &cudaDepthIntrinsics, sizeof(Matrix3f)));
    CUDA_CALL(cudaMemcpy(cudaDepthIntrinsics, depthIntrinsics.data(), sizeof(Matrix3f), cudaMemcpyHostToDevice));


	// We store the estimated camera poses. [on Host memory]
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

	// SurfaceMeasurement object for step 1
    SurfaceMeasurement surfaceMeasurement(depthIntrinsics.inverse(), 0.5, 0.5,  0);

    // Defining memory for previous and current frames,  [on Device memory]
    FrameData previousFrame;
    FrameData currentFrame;

    previousFrame.width =  depthFrameWidth;
    previousFrame.height = depthFrameHeight;

    currentFrame.width =  depthFrameWidth;
    currentFrame.height = depthFrameHeight;

    CUDA_CALL(cudaMalloc((void **) &previousFrame.depthMap, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.g_vertices, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.g_normals, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &previousFrame.globalCameraPose, sizeof(Matrix4f)));

    CUDA_CALL(cudaMalloc((void **) &currentFrame.depthMap, N * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.g_vertices, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.g_normals, N * sizeof(Vector3f)));
    CUDA_CALL(cudaMalloc((void **) &currentFrame.globalCameraPose, sizeof(Matrix4f)));

    CUDA_CALL(cudaMemcpy(previousFrame.globalCameraPose, currentCameraToWorld.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(currentFrame.globalCameraPose, currentCameraToWorld.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

    Matrix4f *tmp4fMat_cpu;
    tmp4fMat_cpu = (Matrix4f*) malloc(sizeof(Matrix4f));

	int i = 0;
	const int iMax = 2;
	while (sensor.processNextFrame() && i < iMax) {
	    // Get current depth frame
		float* depthMap = sensor.getDepth();

		// Copy depth map to current frame, device memory
        CUDA_CALL(cudaMemcpy(currentFrame.depthMap, depthMap, N * sizeof(float), cudaMemcpyHostToDevice));

        // #### Step 1: Surface measurement
        // It expects the pointers for device memory
        surfaceMeasurement.measureSurface(depthFrameWidth, depthFrameHeight,
                                          currentFrame.g_vertices, currentFrame.g_normals, currentFrame.depthMap,
                                          0);


        ///// Debugging code  start
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
		// Don't do ICP on 1st  frame
		if (i > 0) {
            if (USE_GPU_ICP)  {
                // The arguments should be on device memory,expect the initialPose.
                // The returned pose matrix will be on host memory
                currentCameraToWorld = optimizer->estimatePose(*cudaDepthIntrinsics, currentFrame, previousFrame, Matrix4f::Identity());
            }
            else {
                currentCameraToWorld = optimizer->estimatePose(depthIntrinsics, currentFrame, previousFrame, Matrix4f::Identity());
            }
		}

		// Step 3:  Volumetric Grid Fusion

		// Step 4: Ray-Casting

		// Step 5: Update trajectory poses and transform  current points
		// Invert the transformation matrix to get the current camera pose.  [Host memory]
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

        std::cout << "GT camera pose: " << std::endl << sensor.getTrajectory() << std::endl;

		// Update global rotation+translation
		// Copy from device memory to host memory, then update.
        CUDA_CALL(cudaMemcpy(tmp4fMat_cpu, previousFrame.globalCameraPose->data(), sizeof(Matrix4f), cudaMemcpyDeviceToHost));
		*tmp4fMat_cpu = currentCameraPose * *tmp4fMat_cpu;
        CUDA_CALL(cudaMemcpy(currentFrame.globalCameraPose, (*tmp4fMat_cpu).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

        // @TODO: Step 6: Transform all points and normals to new  camera pose
        // IMPORTANT STEP  MISSING

        // Step 7: Update data (e.g. Poses, depth frame etc.) for next frame
		// Update previous frame data
        FrameData tmpFrame = previousFrame;
        previousFrame = currentFrame;
        currentFrame = tmpFrame;

		// if (i % 5 == 0) {
		if (i>0) {
            // We write out the mesh to file for debugging.
            SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
            SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
            SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());

            std::stringstream ss;
            ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
            if (!resultingMesh.writeMesh(ss.str())) {
                std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
                return -1;
            }
		}

		i++;
	}

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

    delete optimizer;

	return 0;
}

//int reconstructRoom() {
//    std::string filenameIn = std::string("../../data/rgbd_dataset_freiburg1_xyz/");
//    std::string filenameBaseOut = std::string("../../outputs/mesh_");
//
//    // Load video
//    std::cout << "Initialize virtual sensor..." << std::endl;
//    VirtualSensor sensor;
//    if (!sensor.init(filenameIn)) {
//        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
//        return -1;
//    }
//
//    // We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
//    sensor.processNextFrame();
//
//    // Setup the optimizer.
//    ICPOptimizer* optimizer = new LinearICPOptimizer();
//    optimizer->setMatchingMaxDistance(0.1f);
//    optimizer->usePointToPlaneConstraints(true);
//    optimizer->setNbOfIterations(20);
//
//    // This will back-project the points to 3D-space and compute the normals
//    // PointCloud target{ depthMap, depthIntrinsics, depthExtrinsics, width, height };
//
//
//    float* depthMap = sensor.getDepth();
//    const Matrix3f& depthIntrinsics = sensor.getDepthIntrinsics();
//    // As we dont know the extrinsics, so setting to identity ????????
//    Matrix4f depthExtrinsics = Matrix4f::Identity(); // sensor.getDepthExtrinsics();
//    const unsigned depthFrameWidth = sensor.getDepthImageWidth();
//    const unsigned depthFrameHeight = sensor.getDepthImageHeight();
//
//    Matrix4f globalCameraPose = Matrix4f::Identity();
//
//    // We store the estimated camera poses.
//    std::vector<Matrix4f> estimatedPoses;
//    Matrix4f currentCameraToWorld = Matrix4f::Identity();
//    estimatedPoses.push_back(currentCameraToWorld.inverse());
//
//    PointCloud* previousFramePC = new PointCloud(depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight );
//
//    int i = 0;
//    const int iMax = 2;
//    while (sensor.processNextFrame() && i <= iMax) {
//        // Get current depth frame
//        float* depthMap = sensor.getDepth();
//
//        // Create a Point Cloud for current frame
//        // We down-sample the source image to speed up the correspondence matching.
//        PointCloud source{ depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight, 8 };
//
//        // Estimate the current camera pose from source to target mesh with ICP optimization.
//        currentCameraToWorld = optimizer->estimatePose(source, *previousFramePC, currentCameraToWorld);
//
//        // Invert the transformation matrix to get the current camera pose.
//        Matrix4f currentCameraPose = currentCameraToWorld.inverse();
//        std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
//        estimatedPoses.push_back(currentCameraPose);
//
//        // update global rotation+translation
//        globalCameraPose = currentCameraPose * globalCameraPose;
//        depthExtrinsics = currentCameraToWorld * depthExtrinsics;
//        // Update previous frame PC
//        delete previousFramePC;
//        previousFramePC = new PointCloud(depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight );
//
//        // if (i % 5 == 0) {
//        if (1) {
//            // We write out the mesh to file for debugging.
//            SimpleMesh currentDepthMesh{ sensor, currentCameraPose, 0.1f };
//            SimpleMesh currentCameraMesh = SimpleMesh::camera(currentCameraPose, 0.0015f);
//            SimpleMesh resultingMesh = SimpleMesh::joinMeshes(currentDepthMesh, currentCameraMesh, Matrix4f::Identity());
//
//            std::stringstream ss;
//            ss << filenameBaseOut << sensor.getCurrentFrameCnt() << ".off";
//            if (!resultingMesh.writeMesh(ss.str())) {
//                std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
//                return -1;
//            }
//        }
//
//        i++;
//    }
//
//    delete optimizer;
//
//    return 0;
//}

int main() {
    int result = reconstructRoom();
	return result;
}
