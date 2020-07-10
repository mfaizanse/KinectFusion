#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

//#include <ceres/ceres.h>
//#include <ceres/rotation.h>
//#include <flann/flann.hpp>

#include "SimpleMesh.h"
//#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "Utils.h"

void getCorrespondencesCPU(
        size_t idx,
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
        const float angleThreshold,
        MatrixXf& A,
        VectorXf& b
) {

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

//            const auto& s = sourcePoints[i];
//            const auto& d = targetPoints[i];
//            const auto& n = targetNormals[i];
//
//            const auto row1 = 4 * i;
//            const auto row2 = 4 * i + 1;
//            const auto row3 = 4 * i + 2;
//            const auto row4 = 4 * i + 3;
//
//            // Add the point-to-plane constraints to the system
//            A(row1,0) = n[2] * s[1] - n[1] * s[2];
//            A(row1,1) = n[0] * s[2] - n[2] * s[0];
//            A(row1,2) = n[1] * s[0] - n[0] * s[1];
//            A(row1,3) = n[0];
//            A(row1,4) = n[1];
//            A(row1,5) = n[2];
//
//            b[row1] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

}

/**
 * ICP optimizer - Abstract Base Class, using Ceres for optimization.
 */
class ICPOptimizer {
public:
    ICPOptimizer() :
            m_bUsePointToPlaneConstraints{ false },
            m_nIterations{ 20 },
            distanceThreshold{ 0.1f },
            angleThreshold{ 1.0472f }
    { }

    void setMatchingMaxDistance(float maxDistance) {
        distanceThreshold = maxDistance;
    }

    void setMatchingMaxAngle(float maxAngle) {
        angleThreshold = maxAngle;
    }

    void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
        m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
    }

    void setNbOfIterations(unsigned nIterations) {
        m_nIterations = nIterations;
    }

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f initialPose = Matrix4f::Identity()) = 0;

    virtual ~ICPOptimizer() {}

protected:
    bool m_bUsePointToPlaneConstraints;
    unsigned m_nIterations;
    float distanceThreshold;
    float angleThreshold;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
        std::vector<Vector3f> transformedPoints;
        transformedPoints.reserve(sourcePoints.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : sourcePoints) {
            transformedPoints.push_back(rotation * point + translation);
        }

        return transformedPoints;
    }

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
        std::vector<Vector3f> transformedNormals;
        transformedNormals.reserve(sourceNormals.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : sourceNormals) {
            transformedNormals.push_back(rotation.inverse().transpose() * normal);
        }

        return transformedNormals;
    }
};

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer {
public:
    LinearICPOptimizer() {}
    ~LinearICPOptimizer() {}

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f initialPose = Matrix4f::Identity()) override {

        size_t N = currentFrame.width * currentFrame.height;

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ... Iteration: " << i << std::endl;
            clock_t begin = clock();

            // Build the system
            MatrixXf A = MatrixXf::Zero(N, 6);
            VectorXf b = VectorXf::Zero(N);

            for (unsigned int v = 0; v < currentFrame.height; ++v) {
                // For every pixel in a row.
                for (unsigned int u = 0; u < currentFrame.width; ++u) {
                    size_t idx = u * currentFrame.width + v;

                    getCorrespondencesCPU(
                            idx,
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
                            angleThreshold,
                            A, b
                    );
                }
            }

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Matching Completed in " << elapsedSecs << " seconds." << std::endl;

            // Solve the system
            VectorXf x(6);
            JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
            x = svd.solve(b);

            float alpha = x(0), beta = x(1), gamma = x(2);

            // Build the pose matrix
            Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

            Vector3f translation = x.tail(3);

            // Build the pose matrix using the rotation and translation matrices
            Matrix4f estimatedPose2 = Matrix4f::Identity();
            estimatedPose2.block(0, 0, 3, 3) = rotation;
            estimatedPose2.block(0, 3, 3, 1) = translation;

            estimatedPose = estimatedPose2 * estimatedPose;

            std::cout << "Optimization iteration done." << std::endl;
        }

        return estimatedPose;
    }

private:
    bool test;
//    Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
//        const unsigned nPoints = sourcePoints.size();
//
//        // Build the system
//        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
//        VectorXf b = VectorXf::Zero(4 * nPoints);
//
//        for (unsigned i = 0; i < nPoints; i++) {
//            const auto& s = sourcePoints[i];
//            const auto& d = targetPoints[i];
//            const auto& n = targetNormals[i];
//
//            const auto row1 = 4 * i;
//            const auto row2 = 4 * i + 1;
//            const auto row3 = 4 * i + 2;
//            const auto row4 = 4 * i + 3;
//
//            // TODO: Add the point-to-plane constraints to the system
//            A(row1,0) = n[2] * s[1] - n[1] * s[2];
//            A(row1,1) = n[0] * s[2] - n[2] * s[0];
//            A(row1,2) = n[1] * s[0] - n[0] * s[1];
//            A(row1,3) = n[0];
//            A(row1,4) = n[1];
//            A(row1,5) = n[2];
//
//            b[row1] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];
//
//            // TODO: Add the point-to-point constraints to the system
//            // x-coordinate
//            RowVectorXf vecrow2(6);
//            vecrow2 << 0, s[2], -s[1], 1, 0, 0;
//            A.row(row2) = vecrow2;
//
//            b[row2] = d[0] - s[0];
//
//            // // y-coordinate
//            RowVectorXf vecrow3(6);
//            vecrow3 << -s[2], 0, s[0], 0, 1, 0;
//            A.row(row3) = vecrow3;
//
//            b[row3] = d[1] - s[1];
//
//            // // z-coordinate
//            RowVectorXf vecrow4(6);
//            vecrow4 << s[1], -s[0], 0, 0, 0, 1;
//            A.row(row4) = vecrow4;
//
//            b[row4] = d[2] - s[2];
//
//            // TODO: Optionally, apply a higher weight to point-to-plane correspondences
//            // A.row(row1) *= 100;
//            // b.row(row1) *= 100;
//        }
//
//        // TODO: Solve the system
//        VectorXf x(6);
//        JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
//        x = svd.solve(b);
//
//        float alpha = x(0), beta = x(1), gamma = x(2);
//
//        // Build the pose matrix
//        Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
//                            AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
//                            AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();
//
//        Vector3f translation = x.tail(3);
//
//        // TODO: Build the pose matrix using the rotation and translation matrices
//        Matrix4f estimatedPose = Matrix4f::Identity();
//        estimatedPose.block(0, 0, 3, 3) = rotation;
//        estimatedPose.block(0, 3, 3, 1) = translation;
//
//        return estimatedPose;
//    }
};