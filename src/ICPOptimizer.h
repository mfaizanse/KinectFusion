#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"


/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
	output[0] = T(input[0]);
	output[1] = T(input[1]);
	output[2] = T(input[2]);
}


/**
 * Pose increment is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 6.
 */
template <typename T>
class PoseIncrement {
public:
	explicit PoseIncrement(T* const array) : m_array{ array } { }
	
	void setZero() {
		for (int i = 0; i < 6; ++i)
			m_array[i] = T(0);
	}

	T* getData() const {
		return m_array;
	}

	/**
	 * Applies the pose increment onto the input point and produces transformed output point.
	 * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
	 * beforehand).
	 */
	void apply(T* inputPoint, T* outputPoint) const {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		const T* rotation = m_array;
		const T* translation = m_array + 3;

		T temp[3];
		ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

		outputPoint[0] = temp[0] + translation[0];
		outputPoint[1] = temp[1] + translation[1];
		outputPoint[2] = temp[2] + translation[2];
	}

	/**
	 * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
	 * transformation 4x4 matrix.
	 */
	static Matrix4f convertToMatrix(const PoseIncrement<double>& poseIncrement) {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		double* pose = poseIncrement.getData();
		double* rotation = pose;
		double* translation = pose + 3;

		// Convert the rotation from SO3 to matrix notation (with column-major storage).
		double rotationMatrix[9];
		ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

		// Create the 4x4 transformation matrix.
		Matrix4f matrix;
		matrix.setIdentity();
		matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
		matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
		matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);
		
		return matrix;
	}

private:
	T* m_array;
};


/**
 * Optimization constraints.
 */
class PointToPointConstraint {
public:
	PointToPointConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) :
		m_sourcePoint{ sourcePoint },
		m_targetPoint{ targetPoint },
		m_weight{ weight }
	{ }

	template <typename T>
	bool operator()(const T* const pose, T* residuals) const {
		// TODO: Implement the point-to-point cost function.
		// The resulting 3D residual should be stored in the residuals array. To apply the pose 
		// increment (pose parameters) to the source point, you can use the PoseIncrement class.
		// Important: Ceres automatically squares the cost function.

		T inputPoint[3];
		T posedPoint[3];

		fillVector(m_sourcePoint, inputPoint);

		auto poseCasted = const_cast<T*>(pose);

		PoseIncrement<T> poseInc(poseCasted);
		poseInc.apply(inputPoint, posedPoint);

        residuals[0] = T(m_weight) * (T(m_targetPoint[0]) - posedPoint[0]);
        residuals[1] = T(m_weight) * (T(m_targetPoint[1]) - posedPoint[1]);
        residuals[2] = T(m_weight) * (T(m_targetPoint[2]) - posedPoint[2]);

		return true;
	}

	static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const float weight) {
		return new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
			new PointToPointConstraint(sourcePoint, targetPoint, weight)
		);
	}

protected:
	const Vector3f m_sourcePoint;
	const Vector3f m_targetPoint;
	const float m_weight;
	const float LAMBDA = 0.1f;
};

class PointToPlaneConstraint {
public:
	PointToPlaneConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) :
		m_sourcePoint{ sourcePoint },
		m_targetPoint{ targetPoint },
		m_targetNormal{ targetNormal },
		m_weight{ weight }
	{ }

	template <typename T>
	bool operator()(const T* const pose, T* residuals) const {
		// TODO: Implement the point-to-plane cost function.
		// The resulting 1D residual should be stored in the residuals array. To apply the pose 
		// increment (pose parameters) to the source point, you can use the PoseIncrement class.
		// Important: Ceres automatically squares the cost function.

    T inputPoint[3];
    T posedPoint[3];
    T targetPoint[3];
    T targetNormal[3];

    fillVector(m_sourcePoint, inputPoint);
    fillVector(m_targetPoint, targetPoint);
    fillVector(m_targetNormal, targetNormal);

    auto poseCasted = const_cast<T*>(pose);

    PoseIncrement<T> poseInc(poseCasted);
    poseInc.apply(inputPoint, posedPoint);

    T r[3];
    for (int i=0; i<3;i++) {
        r[i] = posedPoint[i] - targetPoint[i];
    }

    auto dotProduct = targetNormal[0] * r[0] + targetNormal[1] * r[1] + targetNormal[2] * r[2];

		residuals[0] = T(m_weight) * T(dotProduct);

		return true;
	}

	static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector3f& targetPoint, const Vector3f& targetNormal, const float weight) {
		return new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(
			new PointToPlaneConstraint(sourcePoint, targetPoint, targetNormal, weight)
		);
	}

protected:
	const Vector3f m_sourcePoint;
	const Vector3f m_targetPoint;
	const Vector3f m_targetNormal;
	const float m_weight;
	const float LAMBDA = 1.0f;
};


/**
 * ICP optimizer - Abstract Base Class, using Ceres for optimization.
 */
class ICPOptimizer {
public:
	ICPOptimizer() : 
		m_bUsePointToPlaneConstraints{ false },
		m_nIterations{ 20 },
		m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
	{ }

	void setMatchingMaxDistance(float maxDistance) {
		m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
	}

	void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
		m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
	}

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}

	virtual Matrix4f estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f initialPose = Matrix4f::Identity()) = 0;

    virtual ~ICPOptimizer() {}

protected:
	bool m_bUsePointToPlaneConstraints;
	unsigned m_nIterations;
	std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

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

	void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
		const unsigned nPoints = sourceNormals.size();

		for (unsigned i = 0; i < nPoints; i++) {
			Match& match = matches[i];
			if (match.idx >= 0) {
				const auto& sourceNormal = sourceNormals[i];
				const auto& targetNormal = targetNormals[match.idx];

				// TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60
				float angle = (sourceNormal.dot(targetNormal)) / (sourceNormal.norm() * targetNormal.norm());
                angle = acos(angle);
				//// 60 degrees = 1.0472 radians
				if (angle > 1.0472) {
                    match.idx = -1;
                    matches[i].weight = 0;
				}
			}
		}
	}
};

/**
 * ICP optimizer - using linear least-squares for optimization.
 */
class LinearICPOptimizer : public ICPOptimizer {
public:
	LinearICPOptimizer() {}
	~LinearICPOptimizer() {}

	virtual Matrix4f estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f initialPose = Matrix4f::Identity()) override {
		// Build the index of the FLANN tree (for fast nearest neighbor lookup).
		m_nearestNeighborSearch->buildIndex(target.getPoints());

		// The initial estimate can be given as an argument.
		Matrix4f estimatedPose = initialPose;

		for (int i = 0; i < m_nIterations; ++i) {
			// Compute the matches.
			std::cout << "Matching points ... Iteration: " << i << std::endl;
			clock_t begin = clock();

			auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
			auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
			pruneCorrespondences(transformedNormals, target.getNormals(), matches);

			clock_t end = clock();
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

			std::vector<Vector3f> sourcePoints;
			std::vector<Vector3f> targetPoints;

			// Add all matches to the sourcePoints and targetPoints vectors,
			// so that sourcePoints[i] matches targetPoints[i].
			for (int j = 0; j < transformedPoints.size(); j++) {
				const auto& match = matches[j];
				if (match.idx >= 0) {
					sourcePoints.push_back(transformedPoints[j]);
					targetPoints.push_back(target.getPoints()[match.idx]);
				}
			}

			// Estimate the new pose
			if (m_bUsePointToPlaneConstraints) {
				estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, target.getNormals()) * estimatedPose;
			}
			else {
				estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
			}

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