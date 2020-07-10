#pragma once

#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "Utils.h"
#include "CudaICPOptimizer.h"
#include "ICPOptimizer.h"

#define N_FIXED 640*480

__global__ void getCorrespondences(
        const float *currentDepthMap,
        const Matrix4f *previousGlobalCameraPose,
        const Vector3f *currentVertices,
        const Vector3f *currentNormals,
        const Vector3f *previousVertices,
        const Vector3f *previousNormals,
        const Matrix3f *intrinsics,
        const size_t width,
        const size_t height,
        const size_t N,
        const float distanceThreshold,
        const float angleThreshold,
        Matrix<float, N_FIXED, 6> *A,
        Matrix<float, N_FIXED, 1> *b
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    bool isCorrespondenceFound = false;

    if (currentDepthMap[idx] > 0) {
        // printf("a1\n");
        // Transform previous point to camera coordinates from  world coordinates
        Matrix4f poseInv = (*previousGlobalCameraPose).inverse();
        auto pvH= previousVertices[idx].homogeneous();
        Vector4f r1 =  poseInv * pvH;
        Vector3f v_t_1 = Vector3f(r1.x(), r1.y(), r1.z());
        //Vector3f v_t_1 =  ((*previousGlobalCameraPose).inverse() * previousVertices[idx].homogeneous()).hnormalized();

        // Perspective project to image space
        Vector3f p = *intrinsics * v_t_1;
        int u = (int) (p[0] / p[2]);
        int v = (int) (p[1] / p[2]);

        size_t id2 = u * width + v;

        // check if this point lies in frame and also have a normal
        if(u >= 0 && u < width && v >= 0 &&  v < height && previousNormals[idx].x() != -MINF && currentVertices[id2].x() != -MINF) {
            // printf("a2\n");
            // Get this point p in current frame transform it into world coordinates

            Vector4f cvH = currentVertices[id2].homogeneous();
            Vector4f r2 = *previousGlobalCameraPose * cvH;
            Vector3f v_t = Vector3f(r2.x(), r2.y(), r2.z());

            Matrix3f rotation = previousGlobalCameraPose->block(0,  0, 3, 3);
            Vector3f n_t = rotation * currentNormals[id2];

            // check distance threshold
            float distance = (v_t - previousVertices[idx]).norm();
            // check angle between normals
            float angle = (n_t.dot(previousNormals[idx])) / (n_t.norm() * previousNormals[idx].norm());
            angle = acos(angle);

            if (distance <= distanceThreshold && angle <= angleThreshold) {
                // @TODO: Correct  match, add  this to matches  list
                isCorrespondenceFound = true;

                //printf("Match found  %I - %I \n", idx,  id2);

                Vector3f s = currentVertices[id2];
                Vector3f d = previousVertices[idx];
                Vector3f n = previousNormals[idx];

                // Add the point-to-plane constraints to the system
                (*A)(idx,0) = n[2] * s[1] - n[1] * s[2];
                (*A)(idx,1) = n[0] * s[2] - n[2] * s[0];
                (*A)(idx,2) = n[1] * s[0] - n[0] * s[1];
                (*A)(idx,3) = n[0];
                (*A)(idx,4) = n[1];
                (*A)(idx,5) = n[2];

                (*b)[idx] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];
            }
        }
    }

    if(!isCorrespondenceFound) {
        // printf("Match not found\n");

        (*A)(idx,0) = (*A)(idx,1) = (*A)(idx,2) = (*A)(idx,3) = (*A)(idx,4) = (*A)(idx,5) = 0.0f;
        (*b)[idx] = 0.0f;
    }
}

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPCudaOptimizer : public ICPOptimizer {
public:
    LinearICPCudaOptimizer() {}
    ~LinearICPCudaOptimizer() {}

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f initialPose = Matrix4f::Identity()) override {

        const size_t N = currentFrame.width * currentFrame.height;
        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        Matrix<float, N_FIXED, 6> *A;
        Matrix<float, N_FIXED, 1> *b;

        CUDA_CALL(cudaMalloc((void **) &A, sizeof(Matrix<float, N_FIXED, 6>)));
        CUDA_CALL(cudaMalloc((void **) &b, sizeof(Matrix<float, N_FIXED, 1>)));

        Matrix<float, N_FIXED, 6> *A_cpu;
        Matrix<float, N_FIXED, 1> *b_cpu;

        A_cpu = (Matrix<float, N_FIXED, 6> *) malloc(sizeof(Matrix<float, N_FIXED, 6>));
        b_cpu = (Matrix<float, N_FIXED, 1> *) malloc(sizeof(Matrix<float, N_FIXED, 1>));

//        CUDA_CALL(cudaMemcpy(g_vertices_host, currentFrame.g_vertices, N * sizeof(Vector3f), cudaMemcpyDeviceToHost));

//        CUDA_CALL(cudaMemcpy(A, A_cpu.data(), sizeof(A_cpu), cudaMemcpyHostToDevice));
//        CUDA_CALL(cudaMemcpy(b, b_cpu.data(), sizeof(b_cpu), cudaMemcpyHostToDevice));

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
                    angleThreshold,
                    A,
                    b
                    );

            CUDA_CHECK_ERROR

            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();

            //cudaStreamSynchronize(0);

            clock_t end = clock();
            double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << "Matching Completed in " << elapsedSecs << " seconds." << std::endl;

            // @TODO: Chck if correc,  Shouldn't we do A.data() and b.data()
            CUDA_CALL(cudaMemcpy(A_cpu, A->data(), sizeof(Matrix<float, N_FIXED, 6>), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(b_cpu, b->data(), sizeof(Matrix<float, N_FIXED, 1>), cudaMemcpyDeviceToHost));

            // Solve the system
            VectorXf x(6);
            //std::cout << "estimatedPose-1 "  << std::endl;
            JacobiSVD<MatrixXf> svd(*A_cpu, ComputeThinU | ComputeThinV);
            //std::cout << "estimatedPose-2 "  << std::endl;
            x = svd.solve(*b_cpu);
            //std::cout << "estimatedPose-3 "  << std::endl;

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

            // std::cout << "estimatedPose- " << std::endl << estimatedPose << std::endl;

            std::cout << "Optimization iteration done." << std::endl;
        }

        CUDA_CALL(cudaFree(A));
        CUDA_CALL(cudaFree(b));

        free(A_cpu);
        free(b_cpu);

        return estimatedPose;
    }

private:
    bool test;
};