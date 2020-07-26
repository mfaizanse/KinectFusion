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

// My GPU only supports max 48kb of shared memory per block, so I am using smaller block size
#define BLOCKSIZE_REDUCED 256

__global__ void sumReduction(
        Matrix<float, 6, 6> *AtAs,
        Matrix<float, 6, 1> *Atbs
) {
    // Allocate shared memory
    __shared__ Matrix<float, 6, 6> partial_sum_ata[BLOCKSIZE_REDUCED];
    __shared__ Matrix<float, 6, 1> partial_sum_atb[BLOCKSIZE_REDUCED];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    ///// ##### Tree Reduction (bank conflicts approach)
    ///// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/bank_conflicts/sumReduction.cu
    // Load elements into shared memory
    partial_sum_ata[threadIdx.x] = AtAs[idx];
    partial_sum_atb[threadIdx.x] = Atbs[idx];
    __syncthreads();

    // Increase the stride of the access until we exceed the CTA dimensions
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Change the indexing to be sequential threads
        int index = 2 * s * threadIdx.x;

        // Each thread does work unless the index goes off the block
        if (index < blockDim.x) {
            partial_sum_ata[index] += partial_sum_ata[index + s];
            partial_sum_atb[index] += partial_sum_atb[index + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this block
    if (threadIdx.x == 0) {
        AtAs[blockIdx.x] = partial_sum_ata[0];
        Atbs[blockIdx.x] = partial_sum_atb[0];
    }
}

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
        Matrix<float, 6, 6> *AtAs,
        Matrix<float, 6, 1> *Atbs
) {
    // Allocate shared memory
    __shared__ Matrix<float, 6, 6> partial_sum_ata[BLOCKSIZE_REDUCED];
    __shared__ Matrix<float, 6, 1> partial_sum_atb[BLOCKSIZE_REDUCED];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Terminate all un-necessary threads
    if (idx >= N) {
        printf("WARNING!!!!!!!!!!!!!!!!!: UN-NESSARY THREAD IN ICP, TREE REDUCTION MAY GET STUCK...!!");
        return;
    }

    Matrix<float, 6, 6> local_ata = Matrix<float, 6, 6>::Zero();
    Matrix<float, 6, 1> local_atb = Matrix<float, 6, 1>::Zero();

    //if (currentDepthMap[idx] > 0) {
    if (previousVertices[idx].x() != -MINF) {
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
                // Correct  match, add  this to matches  list
                //isCorrespondenceFound = true;

                Vector3f s = currentVertices[id2];
                Vector3f d = previousVertices[idx];
                Vector3f n = previousNormals[idx];

                Matrix<float,6,1> at;

                // Add the point-to-plane constraints to the system
                auto t1 = s.cross(n);
                at(0) = t1[0];
                at(1) = t1[1];
                at(2) = t1[2];
//                at(0) = n[2] * s[1] - n[1] * s[2];
//                at(1) = n[0] * s[2] - n[2] * s[0];
//                at(2) = n[1] * s[0] - n[0] * s[1];
                at(3) = n[0];
                at(4) = n[1];
                at(5) = n[2];

                float b = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

                local_ata = at * at.transpose();
                local_atb = at * b;
            }
        }
    }

    ///// ##### Tree Reduction (bank conflicts approach)
    ///// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/sumReduction/bank_conflicts/sumReduction.cu
    // Load elements into shared memory
    partial_sum_ata[threadIdx.x] = local_ata;
    partial_sum_atb[threadIdx.x] = local_atb;
    __syncthreads();

    // Increase the stride of the access until we exceed the CTA dimensions
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Change the indexing to be sequential threads
        int index = 2 * s * threadIdx.x;

        // Each thread does work unless the index goes off the block
        if (index < blockDim.x) {
            partial_sum_ata[index] += partial_sum_ata[index + s];
            partial_sum_atb[index] += partial_sum_atb[index + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is indexed by this block
    if (threadIdx.x == 0) {
        AtAs[blockIdx.x] = partial_sum_ata[0];
        Atbs[blockIdx.x] = partial_sum_atb[0];
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
    LinearICPCudaOptimizer(size_t width, size_t height, cudaStream_t stream = 0) {
        this->stream = stream;
        const size_t N = width * height;

        CUDA_CALL(cudaMalloc((void **) &estimatedPose, sizeof(Matrix4f)));
        CUDA_CALL(cudaMalloc((void**) &atas, sizeof(Matrix<float,6,6>) * N));
        CUDA_CALL(cudaMalloc((void**) &atbs, sizeof(Matrix<float,6,1>) * N));

        CUDA_CALL(cudaMalloc((void **) &transformedVertices, N * sizeof(Vector3f)));
        CUDA_CALL(cudaMalloc((void **) &transformedNormals, N * sizeof(Vector3f)));
    }
    ~LinearICPCudaOptimizer() {
        CUDA_CALL(cudaFree(estimatedPose));
        CUDA_CALL(cudaFree(atas));
        CUDA_CALL(cudaFree(atbs));

        CUDA_CALL(cudaFree(transformedVertices));
        CUDA_CALL(cudaFree(transformedNormals));
    }

    virtual Matrix4f estimatePose(Matrix3f& intrinsics, const FrameData& currentFrame, const FrameData& previousFrame, Matrix4f& initialPose) override {
        const size_t N = currentFrame.width * currentFrame.height;

        // The initial estimate can be given as an argument.
        CUDA_CALL(cudaMemcpy(estimatedPose, initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToDevice));

        Matrix4f estimatedPose_cpu;
        CUDA_CALL(cudaMemcpy(estimatedPose_cpu.data(), initialPose.data(), sizeof(Matrix4f), cudaMemcpyDeviceToHost));

        clock_t begin = clock();

        for (int i = 0; i < m_nIterations; ++i) {
            // Compute the matches.
            //std::cout << "Matching points ... Iteration: " << i << std::endl;

            // Transform points and normals.  IMPORTANT.
            // 640*480 = 307200
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

            // 1200 blocks
            getCorrespondences<<<(N + BLOCKSIZE_REDUCED - 1) / BLOCKSIZE_REDUCED, BLOCKSIZE_REDUCED, 0, 0 >>> (
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
                    atas,
                    atbs
                    );

            CUDA_CHECK_ERROR

            sumReduction<<<6, 200, 0, 0 >>> (
                    atas,
                    atbs
            );

            CUDA_CHECK_ERROR

            sumReduction<<<1, 6, 0, 0 >>> (
                    atas,
                    atbs
            );

            CUDA_CHECK_ERROR

            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();

            Matrix<float,6,6> ata_cpu = Matrix<float,6,6>::Zero();
            Matrix<float,6,1> atb_cpu = Matrix<float,6,1>::Zero();

            CUDA_CALL(cudaMemcpyAsync(ata_cpu.data(),atas[0].data(),sizeof(Matrix<float,6,6>),cudaMemcpyDeviceToHost,stream));
            CUDA_CALL(cudaMemcpyAsync(atb_cpu.data(),atbs[0].data(),sizeof(Matrix<float,6,1>),cudaMemcpyDeviceToHost,stream));

            VectorXf x(6);
            //x = ata_cpu.triangularView<Upper>().solve(atb_cpu);

            JacobiSVD<MatrixXf> svd(ata_cpu, ComputeThinU | ComputeThinV);
            x = svd.solve(atb_cpu);
            //x = ata_cpu.llt().solve(atb_cpu);
            //x = ata_cpu.llt().matrixLLT().triangularView<StrictlyUpper>().solve(atb_cpu);

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

            estimatedPose_cpu = estimatedPose2 * estimatedPose_cpu;
            CUDA_CALL(cudaMemcpy(estimatedPose, estimatedPose_cpu.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

            // std::cout << "estimatedPose- " << std::endl << estimatedPose << std::endl;

            //std::cout << "Optimization iteration done." << std::endl;
        }

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "ICP " << m_nIterations << " iterations completed in " << elapsedSecs << " seconds." << std::endl;

        return estimatedPose_cpu;
    }

private:
    Matrix4f *estimatedPose; // On device memory
    Matrix<float,6,6> *atas; // On device memory
    Matrix<float,6,1> *atbs; // On device memory
    Vector3f *transformedVertices; // On device memory
    Vector3f *transformedNormals;  // On device memory
    cudaStream_t stream;
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
                            Matrix<float,6,6> *ata,
                            Matrix<float,6,1> *atb) {
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

                Matrix<float,6,1> at;

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

                float b = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] - n[0] * s[0] - n[1] * s[1] - n[2] * s[2];

                ata[idx] = at * at.transpose();
                atb[idx] = at * b;

//                at[0] = s.z() * n.y() - s.y() * n.z();
//                at[1] = s.x() * n.z() - s.z() * n.x();
//                at[2] = s.y() * n.x() - s.x() * n.y();
//                at[3] = n.x();
//                at[4] = n.y();
//                at[5] = n.z();
//
//                float b = n.dot(d - s);
            } else {
                ata[idx] = Matrix<float,6,6>::Zero();
                atb[idx] = Matrix<float,6,1>::Zero();
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
        CUDA_CALL(cudaMalloc((void**) &ata, sizeof(Matrix<float,6,6>) * (N+1)));
        CUDA_CALL(cudaMalloc((void**) &atb, sizeof(Matrix<float,6,1>) * (N+1)));

        CUDA_CALL(cudaMalloc((void **) &estimatedPose, sizeof(Matrix4f)));

        CUDA_CALL(cudaMalloc((void **) &transformedVertices, N * sizeof(Vector3f)));
        CUDA_CALL(cudaMalloc((void **) &transformedNormals, N * sizeof(Vector3f)));

        //Set up cub temp memory beforehand, should be the same for every frame
        cub::DeviceReduce::Reduce(d_temp_storage_ata, temp_storage_bytes_ata, ata, ata + N, N, customAdd, Matrix<float,6,6>::Zero(), stream);
        cub::DeviceReduce::Reduce(d_temp_storage_atb, temp_storage_bytes_atb, atb, atb + N, N, customAdd, Matrix<float,6,1>::Zero(), stream);

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

            cub::DeviceReduce::Reduce(d_temp_storage_ata, temp_storage_bytes_ata, ata, ata + N, N, customAdd, Matrix<float,6,6>::Zero(), stream);
            cub::DeviceReduce::Reduce(d_temp_storage_atb, temp_storage_bytes_atb, atb, atb + N, N, customAdd, Matrix<float,6,1>::Zero(), stream);

            CUDA_CHECK_ERROR

            cudaDeviceSynchronize();

            Matrix<float,6,6> ata_cpu = Matrix<float,6,6>::Zero();
            Matrix<float,6,1> atb_cpu = Matrix<float,6,1>::Zero();

            CUDA_CALL(cudaMemcpyAsync(&ata_cpu,(ata + N)->data(),sizeof(Matrix<float,6,6>),cudaMemcpyDeviceToHost,stream));
            CUDA_CALL(cudaMemcpyAsync(&atb_cpu,(atb + N)->data(),sizeof(Matrix<float,6,1>),cudaMemcpyDeviceToHost,stream));

            cudaDeviceSynchronize();

            VectorXf x(6);

            //JacobiSVD<MatrixXd> svd(ata_cpu, ComputeThinU | ComputeThinV);
            //x = svd.solve(atb_cpu);

            x = ata_cpu.triangularView<Upper>().solve(atb_cpu);

            //x = ata_cpu.llt().solve(atb_cpu);
            //x = ata_cpu.llt().matrixLLT().triangularView<StrictlyUpper>().solve(atb_cpu);

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

            estimatedPose_cpu = estimatedPose2 * estimatedPose_cpu;
            CUDA_CALL(cudaMemcpy(estimatedPose, estimatedPose_cpu.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

            std::cout << "Optimization iteration done." << std::endl;
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
    Matrix<float,6,6> *ata;
    Matrix<float,6,1> *atb;
    Matrix4f *estimatedPose;
    Matrix4f estimatedPose_cpu;
    Vector3f *transformedVertices; // On device memory
    Vector3f *transformedNormals;  // On device memory

    void *d_temp_storage_ata = NULL;
    size_t temp_storage_bytes_ata = 0;
    void *d_temp_storage_atb = NULL;
    size_t temp_storage_bytes_atb = 0;
};