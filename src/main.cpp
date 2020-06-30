#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"

#define SHOW_BUNNY_CORRESPONDENCES 0

#define USE_POINT_TO_PLANE	1
#define USE_LINEAR_ICP		1
#define RUN_SEQUENCE_ICP	1

//void debugCorrespondenceMatching() {
//	// Load the source and target mesh.
//	const std::string filenameSource = std::string("../data/bunny/bunny_part2_trans.off");
//	const std::string filenameTarget = std::string("../data/bunny/bunny_part1.off");
//
//	SimpleMesh sourceMesh;
//	if (!sourceMesh.loadMesh(filenameSource)) {
//		std::cout << "Mesh file wasn't read successfully." << std::endl;
//		return;
//	}
//
//	SimpleMesh targetMesh;
//	if (!targetMesh.loadMesh(filenameTarget)) {
//		std::cout << "Mesh file wasn't read successfully." << std::endl;
//		return;
//	}
//
//	PointCloud source{ sourceMesh };
//	PointCloud target{ targetMesh };
//
//	// Search for matches using FLANN.
//	std::unique_ptr<NearestNeighborSearch> nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
//	nearestNeighborSearch->setMatchingMaxDistance(0.0001f);
//	nearestNeighborSearch->buildIndex(target.getPoints());
//	auto matches = nearestNeighborSearch->queryMatches(source.getPoints());
//
//	// Visualize the correspondences with lines.
//	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, Matrix4f::Identity());
//	auto sourcePoints = source.getPoints();
//	auto targetPoints = target.getPoints();
//
//	for (unsigned i = 0; i < 100; ++i) { // sourcePoints.size()
//		const auto match = matches[i];
//		if (match.idx >= 0) {
//			const auto& sourcePoint = sourcePoints[i];
//			const auto& targetPoint = targetPoints[match.idx];
//			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::cylinder(sourcePoint, targetPoint, 0.002f, 2, 15), resultingMesh, Matrix4f::Identity());
//		}
//	}
//
//	resultingMesh.writeMesh(std::string("../output/correspondences.off"));
//}

int reconstructRoom() {
	std::string filenameIn = std::string("../../data/rgbd_dataset_freiburg1_xyz/");
	std::string filenameBaseOut = std::string("../../outputs/mesh_");

	// Load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.init(filenameIn)) {
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// We store a first frame as a reference frame. All next frames are tracked relatively to the first frame.
	sensor.processNextFrame();

    // Setup the optimizer.
    ICPOptimizer* optimizer = new LinearICPOptimizer();
    optimizer->setMatchingMaxDistance(0.1f);
    optimizer->usePointToPlaneConstraints(true);
    optimizer->setNbOfIterations(20);

    // This will back-project the points to 3D-space and compute the normals
    // PointCloud target{ depthMap, depthIntrinsics, depthExtrinsics, width, height };


    float* depthMap = sensor.getDepth();
    const Matrix3f& depthIntrinsics = sensor.getDepthIntrinsics();
    // As we dont know the extrinsics, so setting to identity ????????
    Matrix4f depthExtrinsics = Matrix4f::Identity(); // sensor.getDepthExtrinsics();
    const unsigned depthFrameWidth = sensor.getDepthImageWidth();
    const unsigned depthFrameHeight = sensor.getDepthImageHeight();

    Matrix4f globalCameraPose = Matrix4f::Identity();

	// We store the estimated camera poses.
	std::vector<Matrix4f> estimatedPoses;
	Matrix4f currentCameraToWorld = Matrix4f::Identity();
	estimatedPoses.push_back(currentCameraToWorld.inverse());

    PointCloud* previousFramePC = new PointCloud(depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight );

	int i = 0;
	const int iMax = 2;
	while (sensor.processNextFrame() && i <= iMax) {
	    // Get current depth frame
		float* depthMap = sensor.getDepth();

		// Create a Point Cloud for current frame
		// We down-sample the source image to speed up the correspondence matching.
		PointCloud source{ depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight, 8 };

        // Estimate the current camera pose from source to target mesh with ICP optimization.
		currentCameraToWorld = optimizer->estimatePose(source, *previousFramePC, currentCameraToWorld);
		
		// Invert the transformation matrix to get the current camera pose.
		Matrix4f currentCameraPose = currentCameraToWorld.inverse();
		std::cout << "Current camera pose: " << std::endl << currentCameraPose << std::endl;
		estimatedPoses.push_back(currentCameraPose);

		// update global rotation+translation
        globalCameraPose = currentCameraPose * globalCameraPose;
        depthExtrinsics = currentCameraToWorld * depthExtrinsics;
		// Update previous frame PC
        delete previousFramePC;
        previousFramePC = new PointCloud(depthMap, depthIntrinsics, depthExtrinsics, depthFrameWidth, depthFrameHeight );

		// if (i % 5 == 0) {
		if (1) {
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

	delete optimizer;

	return 0;
}

int main() {
    int result = reconstructRoom();
	return result;
}
