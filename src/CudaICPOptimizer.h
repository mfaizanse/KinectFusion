#pragma once

#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "Utils.h"
#include "CudaICPOptimizer.h"
#include "ICPOptimizer.h"

__global__ void getCorrespondences(
        float *currentDepthMap,
        Matrix4f *previousGlobalCameraPose,
        Vector3f *currentVertices,
        Vector3f *currentNormals,
        Vector3f *previousVertices,
        Vector3f *previousNormals,
        Matrix3f *intrinsics,
        const size_t width,
        const size_t height,
        const size_t N,
        const float distanceThreshold,
        const float angleThreshold
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    if (currentDepthMap[idx] <= 0) {
        return;
    }

    // Transform previous point to camera coordinates from  world coordinates
    Vector3f v_t_1 =  (previousGlobalCameraPose->inverse() * previousVertices[idx].homogeneous()).hnormalized();

    // Perspective project to image space
    Vector3f p = *intrinsics * v_t_1;
    int u = (int) (p[0] / p[2]);
    int v = (int) (p[1] / p[2]);

    // check if this point lies in frame
    if(u < 0 || u > width - 1 || v  < 0 ||  v > height - 1) {
        return;
    }

    // Get this point p in current frame transform it into world coordinates
    size_t id2 = u * width + v;
    Vector3f v_t = (*previousGlobalCameraPose * currentVertices[id2].homogeneous()).hnormalized();

    Matrix3f rotation = previousGlobalCameraPose->block(0,  0, 3, 3);
    Vector3f n_t = rotation * currentNormals[id2];

    // check distance threshold
    float distance = (v_t - previousVertices[idx]).norm();
    if (distance > distanceThreshold) {
        return;
    }

    // check angle between normals
    float angle = (n_t.dot(previousNormals[idx])) / (n_t.norm() * previousNormals[idx].norm());
    angle = acos(angle);

    if (angle > angleThreshold) {
        return;
    }

    // @TODO: Correct  match, add  this to matches  list

}

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPCudaOptimizer : public ICPOptimizer {
public:
    LinearICPCudaOptimizer() {}
    ~LinearICPCudaOptimizer() {}

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f initialPose = Matrix4f::Identity()) override {

        size_t N = currentFrame.width * currentFrame.height;

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ... Iteration: " << i << std::endl;
            clock_t begin = clock();

            getCorrespondences<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, 0 >>> (
                    currentFrame.depthMap,
                    previousFrame.globalCameraPose,
                    currentFrame.g_vertices,
                    currentFrame.g_normals,
                    previousFrame.g_vertices,
                    previousFrame.g_normals,
                    &intrinsics,
                    currentFrame.width,
                    currentFrame.height,
                    N,
                    distanceThreshold,
                    angleThreshold
                    );

            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();

            CUDA_CHECK_ERROR

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Matching Completed in " << elapsedSecs << " seconds." << std::endl;

//            auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
//            auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);
//
//            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
//            pruneCorrespondences(transformedNormals, target.getNormals(), matches);
//
//            clock_t end = clock();
//            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
//            std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;
//
//            std::vector<Vector3f> sourcePoints;
//            std::vector<Vector3f> targetPoints;
//
//            // Add all matches to the sourcePoints and targetPoints vectors,
//            // so that sourcePoints[i] matches targetPoints[i].
//            for (int j = 0; j < transformedPoints.size(); j++) {
//                const auto& match = matches[j];
//                if (match.idx >= 0) {
//                    sourcePoints.push_back(transformedPoints[j]);
//                    targetPoints.push_back(target.getPoints()[match.idx]);
//                }
//            }
//
//            // Estimate the new pose
//            if (m_bUsePointToPlaneConstraints) {
//                estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
//            }
//            else {
//                estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
//            }

            std::cout << "Optimization iteration done." << std::endl;
        }

        return estimatedPose;
    }

private:
    Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
        ProcrustesAligner procrustAligner;
        Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

        return estimatedPose;
    }

    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
        const unsigned nPoints = sourcePoints.size();

        // Build the system
        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        for (unsigned i = 0; i < nPoints; i++) {
            const auto& s = sourcePoints[i];
            const auto& d = targetPoints[i];
            const auto& n = targetNormals[i];

            const auto row1 = 4 * i;
            const auto row2 = 4 * i + 1;
            const auto row3 = 4 * i + 2;
            const auto row4 = 4 * i + 3;

            // TODO: Add the point-to-plane constraints to the system
            A(row1,0) = n[2] * s[1] - n[1] * s[2];
            A(row1,1) = n[0] * s[2] - n[2] * s[0];
            A(row1,2) = n[1] * s[0] - n[0] * s[1];
            A(row1,3) = n[0];
            A(row1,4) = n[1];
            A(row1,5) = n[2];

            b[row1] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

            // TODO: Add the point-to-point constraints to the system
            // x-coordinate
            RowVectorXf vecrow2(6);
            vecrow2 << 0, s[2], -s[1], 1, 0, 0;
            A.row(row2) = vecrow2;

            b[row2] = d[0] - s[0];

            // // y-coordinate
            RowVectorXf vecrow3(6);
            vecrow3 << -s[2], 0, s[0], 0, 1, 0;
            A.row(row3) = vecrow3;

            b[row3] = d[1] - s[1];

            // // z-coordinate
            RowVectorXf vecrow4(6);
            vecrow4 << s[1], -s[0], 0, 0, 0, 1;
            A.row(row4) = vecrow4;

            b[row4] = d[2] - s[2];

            // TODO: Optionally, apply a higher weight to point-to-plane correspondences
            // A.row(row1) *= 100;
            // b.row(row1) *= 100;
        }

        // TODO: Solve the system
        VectorXf x(6);
        JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
        x = svd.solve(b);

        float alpha = x(0), beta = x(1), gamma = x(2);

        // Build the pose matrix
        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                            AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                            AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

        Vector3f translation = x.tail(3);

        // TODO: Build the pose matrix using the rotation and translation matrices
        Matrix4f estimatedPose = Matrix4f::Identity();
        estimatedPose.block(0, 0, 3, 3) = rotation;
        estimatedPose.block(0, 3, 3, 1) = translation;

        return estimatedPose;
    }
};