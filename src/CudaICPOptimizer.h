#pragma once

#include "SimpleMesh.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "Utils.h"
#include "CudaICPOptimizer.h"
#include "ICPOptimizer.h"
#include <cub/device/device_reduce.cuh>
#include <Eigen/Cholesky>

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

__global__ void transformVerticesAndNormals(
        const Vector3f *vertices,
        const Vector3f *normals,
        const Matrix4f *pose,
        const size_t width,
        const size_t height,
        const size_t N,
        Vector3f *transformedVertices,
        Vector3f *transformedNormals
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

    if (vertices[idx].x() != -MINF) {
        // printf("a1\n");
        auto vH = vertices[idx].homogeneous();
        Vector4f r1 =  *pose * vH;

        transformedVertices[idx].x() = r1.x();
        transformedVertices[idx].y() = r1.y();
        transformedVertices[idx].z() = r1.z();
    }
    else {
        transformedVertices[idx].x() = -MINF;
        transformedVertices[idx].y() = -MINF;
        transformedVertices[idx].z() = -MINF;
    }

    if (normals[idx].x() != -MINF) {
        // printf("a2\n");
        Matrix3f rotation = (*pose).block(0, 0, 3, 3);
        Matrix3f rt =  rotation.inverse();
        rt = rt.transpose();
        Vector3f n2 = rt * normals[idx];

        transformedNormals[idx].x() = n2.x();
        transformedNormals[idx].y() = n2.y();
        transformedNormals[idx].z() = n2.z();
    }
    else {
        transformedNormals[idx].x() = -MINF;
        transformedNormals[idx].y() = -MINF;
        transformedNormals[idx].z() = -MINF;
    }
}

class TransformHelper {
public:
    TransformHelper() {}
    ~TransformHelper() {}

    void transformCurrentFrameVertices(FrameData& currentFrame, const Matrix4f* pose) {
        transformVerticesAndNormals<<<(N_FIXED + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, 0 >>> (
                currentFrame.g_vertices,
                currentFrame.g_normals,
                pose,
                currentFrame.width,
                currentFrame.height,
                N_FIXED,
                currentFrame.g_vertices,
                currentFrame.g_normals
        );

        CUDA_CHECK_ERROR

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();
    }
};


/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPCudaOptimizer : public ICPOptimizer {
public:
    LinearICPCudaOptimizer() {}
    ~LinearICPCudaOptimizer() {}

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f& initialPose) override {

        const size_t N = currentFrame.width * currentFrame.height;
        // The initial estimate can be given as an argument.
        Matrix4f *estimatedPose;
        CUDA_CALL(cudaMalloc((void **) &estimatedPose, sizeof(Matrix4f)));
        CUDA_CALL(cudaMemcpy(estimatedPose, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToDevice));

        Matrix4f *estimatedPose_cpu;
        estimatedPose_cpu = (Matrix4f*) malloc(sizeof(Matrix4f));
        CUDA_CALL(cudaMemcpy(estimatedPose_cpu, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToHost));

        Vector3f *transformedVertices; // On device memory
        Vector3f *transformedNormals;  // On device memory
        CUDA_CALL(cudaMalloc((void **) &transformedVertices, N * sizeof(Vector3f)));
        CUDA_CALL(cudaMalloc((void **) &transformedNormals, N * sizeof(Vector3f)));

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

            // @TODO: Transform points and normals.  IMPORTANT
            transformVerticesAndNormals<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, 0 >>> (
                    currentFrame.g_vertices,
                    currentFrame.g_normals,
                    estimatedPose,
                    currentFrame.width,
                    currentFrame.height,
                    N,
                    transformedVertices,
                    transformedNormals
            );

            CUDA_CHECK_ERROR

            getCorrespondences<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, 0 >>> (
                    currentFrame.depthMap,
                    previousFrame.globalCameraPose,
                    transformedVertices,
                    transformedNormals,
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

            *estimatedPose_cpu = estimatedPose2 * *estimatedPose_cpu;
            CUDA_CALL(cudaMemcpy(estimatedPose, (*estimatedPose_cpu).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

            // std::cout << "estimatedPose- " << std::endl << estimatedPose << std::endl;

            std::cout << "Optimization iteration done." << std::endl;
        }

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

__global__ void computeAtbs(const float *currentDepthMap,
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
                            Matrix<double,6,6> *ata,
                            Matrix<double,6,1> *atb) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        return;
    }

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

                Matrix<double,6,1> at;

                //printf("Match found  %I - %I \n", idx,  id2);

                Vector3f s = currentVertices[id2];
                Vector3f d = previousVertices[idx];
                Vector3f n = previousNormals[idx];

                // Add the point-to-plane constraints to the system
                at(1) = n[0] * s[2] - n[2] * s[0];
                at(0) = n[2] * s[1] - n[1] * s[2];
                at(2) = n[1] * s[0] - n[0] * s[1];
                at(3) = n[0];
                at(4) = n[1];
                at(5) = n[2];

                double b = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

//                at[0] = s.z() * n.y() - s.y() * n.z();
//                at[1] = s.x() * n.z() - s.z() * n.x();
//                at[2] = s.y() * n.x() - s.x() * n.y();
//                at[3] = n.x();
//                at[4] = n.y();
//                at[5] = n.z();
//
//                double b = n.dot(d - s);

                ata[idx] = at * at.transpose();

                atb[idx] = at * b;
            } else {
                ata[idx] = Matrix<double,6,6>::Zero();
                atb[idx] = Matrix<double,6,1>::Zero();
            }
        }
    }
}

struct CustomAdd
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return b + a;
    }
};

class LinearICPCubOptimizer : public ICPOptimizer {
public:
    LinearICPCubOptimizer(size_t width, size_t height, cudaStream_t stream = 0) {
        const size_t N = width * height;

        this->stream = stream;

        //Allocate for temporary results, that get reduced
        CUDA_CALL(cudaMalloc((void**) &ata, sizeof(Matrix<double,6,6>) * (N+1)));
        CUDA_CALL(cudaMalloc((void**) &atb, sizeof(Matrix<double,6,1>) * (N+1)));

        CUDA_CALL(cudaMalloc((void **) &estimatedPose, sizeof(Matrix4f)));

        CUDA_CALL(cudaMalloc((void **) &transformedVertices, N * sizeof(Vector3f)));
        CUDA_CALL(cudaMalloc((void **) &transformedNormals, N * sizeof(Vector3f)));

        //Set up cub temp memory beforehand, should be the same for every frame
        cub::DeviceReduce::Reduce(d_temp_storage_ata, temp_storage_bytes_ata, ata, ata + N, N, customAdd, Matrix<double,6,6>::Zero(), stream);
        cub::DeviceReduce::Reduce(d_temp_storage_atb, temp_storage_bytes_atb, atb, atb + N, N, customAdd, Matrix<double,6,1>::Zero(), stream);

        std::cout << "temp_storage_bytes_ata: " << temp_storage_bytes_ata << std::endl;
        std::cout << "temp_storage_bytes_atb: " << temp_storage_bytes_atb << std::endl;

        CUDA_CALL(cudaMalloc(&d_temp_storage_ata, temp_storage_bytes_ata));
        CUDA_CALL(cudaMalloc(&d_temp_storage_atb, temp_storage_bytes_atb));
    }

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f& initialPose) override {

        const size_t N = currentFrame.width * currentFrame.height;

        CUDA_CALL(cudaMemcpyAsync(estimatedPose, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToDevice, stream));

        CUDA_CALL(cudaMemcpyAsync(estimatedPose_cpu.data(), initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToHost, stream));


        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            std::cout << "Matching points ... Iteration: " << i << std::endl;

            transformVerticesAndNormals<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>>(
                    currentFrame.g_vertices,
                    currentFrame.g_normals,
                    estimatedPose,
                    currentFrame.width,
                    currentFrame.height,
                    N,
                    transformedVertices,
                    transformedNormals
            );

            CUDA_CHECK_ERROR

            computeAtbs<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0, stream >>>(
                    currentFrame.depthMap,
                    previousFrame.globalCameraPose,
                    transformedVertices,
                    transformedNormals,
                    previousFrame.g_vertices,
                    previousFrame.g_normals,
                    &intrinsics,
                    currentFrame.width,
                    currentFrame.height,
                    N,
                    distanceThreshold,
                    angleThreshold,
                    ata,
                    atb
            );

            CUDA_CHECK_ERROR

            cub::DeviceReduce::Reduce(d_temp_storage_ata, temp_storage_bytes_ata, ata, ata + N, N, customAdd, Matrix<double,6,6>::Zero(), stream);
            cub::DeviceReduce::Reduce(d_temp_storage_atb, temp_storage_bytes_atb, atb, atb + N, N, customAdd, Matrix<double,6,1>::Zero(), stream);

            CUDA_CHECK_ERROR

            cudaDeviceSynchronize();

            Matrix<double,6,6> ata_cpu = Matrix<double,6,6>::Zero();
            Matrix<double,6,1> atb_cpu = Matrix<double,6,1>::Zero();

            CUDA_CALL(cudaMemcpyAsync(&ata_cpu,ata + N,sizeof(Matrix<double,6,6>),cudaMemcpyDeviceToHost,stream));
            CUDA_CALL(cudaMemcpyAsync(&atb_cpu,atb + N,sizeof(Matrix<double,6,1>),cudaMemcpyDeviceToHost,stream));



            VectorXd x(6);

            //JacobiSVD<MatrixXd> svd(ata_cpu, ComputeThinU | ComputeThinV);
            //x = svd.solve(atb_cpu);

            x = ata_cpu.triangularView<Upper>().solve(atb_cpu);

            //x = ata_cpu.llt().solve(atb_cpu);

            //x = ata_cpu.llt().matrixLLT().triangularView<StrictlyUpper>().solve(atb_cpu);

            float alpha = x(2), beta = x(0), gamma = x(1);

            // Build the pose matrix
            Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

            Vector3f translation = x.tail(3).cast<float>();

            // Build the pose matrix using the rotation and translation matrices
            Matrix4f estimatedPose2 = Matrix4f::Identity();
            estimatedPose2.block(0, 0, 3, 3) = rotation;
            estimatedPose2.block(0, 3, 3, 1) = translation;

            estimatedPose_cpu = estimatedPose2 * estimatedPose_cpu;
            CUDA_CALL(cudaMemcpy(estimatedPose, estimatedPose_cpu.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

            std::cout << "Solution vector:\n" << x << std::endl;
            std::cout << "AtA:\n" << ata_cpu << std::endl;
            std::cout << "Atb:\n" << atb_cpu << std::endl;
            std::cout << "Optimization iteration done.\nNew estimated Pose:\n" << estimatedPose_cpu << std::endl;
        }

        return estimatedPose_cpu;
    }

    ~LinearICPCubOptimizer() {
        //Free temp memory
        CUDA_CALL(cudaFree(ata));
        CUDA_CALL(cudaFree(atb));
        CUDA_CALL(cudaFree(estimatedPose));
        CUDA_CALL(cudaFree(transformedVertices));
        CUDA_CALL(cudaFree(transformedNormals));
        CUDA_CALL(cudaFree(d_temp_storage_ata));
        CUDA_CALL(cudaFree(d_temp_storage_atb));
    }

private:
    cudaStream_t stream;
    CustomAdd customAdd;
    Matrix<double,6,6> *ata;
    Matrix<double,6,1> *atb;
    Matrix4f *estimatedPose;
    Matrix4f estimatedPose_cpu;
    Vector3f *transformedVertices; // On device memory
    Vector3f *transformedNormals;  // On device memory

    void *d_temp_storage_ata = NULL;
    size_t temp_storage_bytes_ata = 0;
    void *d_temp_storage_atb = NULL;
    size_t temp_storage_bytes_atb = 0;

};