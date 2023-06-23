// Non linear solver, including EdgeDirectVO's GN and LM of us.
// 09/12/2022
// zyao@ncut.edu.cn
#pragma once
#ifndef NLSOLVER_H
#define NLSOLVER_H

// Standard
#include <iostream>
#include <fstream>

// OpenCV
#include <opencv2/calib3d/calib3d.hpp>

// Eigen
#include <Eigen/Core>

// Local
#include <oneframe.h>
#include <tinterval.h>

// Weighting via DVO Student T distribution.
#include <dvo/core/weight_calculation.h>

#include <pose.h>

namespace GGMVO{
	// infinity.
	const float INF_F(std::numeric_limits<float>::infinity());

	// EdgeDirectVO constants.
	const float HUBER_THRESH = 5.f;
	const float MIN_GRADIENT_THRESH = 50.f;
	const double MIN_TRANSLATION_UPDATE(1.e-8);
	const double MIN_ROTATION_UPDATE(1.e-8);
	const double EPSILON(1.e-8);
	const int MAX_ITERATIONS_PER_PYRAMID[] = { 15, 20 }; 
	const int FIRST_ITERATION_ONLEVEL = 0;
	const int OUTPUT_PRECISION(7);

class NLSolver{
    public:

		// Tructors.
        NLSolver();
        ~NLSolver();
		NLSolver(const int & PYRAMID_LEVEL);

        // Give GN of EdgeDirecVO or LM of us.
		void VOviaRGBD();
		void SolverGN(const int level, Eigen::Matrix<double, 6, Eigen::RowMajor>& poseupdate);
		void SolverLM(const int level, Eigen::Matrix<double, 6, Eigen::RowMajor>& poseupdate);

		void calculateRef3D();
        void prepareVectors(int level);
        void make3DPoints(const cv::Mat& cameraMatrix, int level);
        float warpAndProject(const Eigen::Matrix<double,4,4>& invPose, int level);
        float interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const;
        bool checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel);

		// Trajectory operations
		void writePose(const Pose& pose, const std::string & timestamp);
		const cv::Mat getLastRelativePose() const;
		const Pose& getCurrentPose() const;
		void addPose(Pose& newPose);
		void advanceFrameStream();

		// Configure WeightCalculation.
		void configSTDweight(const ScaleEstimators::enum_t & scaleType);
		inline void computeScale(const cv::Mat& residuals, const int & currentLevel, const int & currentIterationOnLevel);
		float warpAndProject(const Eigen::Matrix<double, 4, 4>& invPose, int level, const int & iterOnlevel); // overload it.

    private:        
		// Frame things.
		OneFrame * mOF_refFrame;
		OneFrame * mOF_curFrame;
		int mint_refFrameIndex;
		int mint_curFrameIndex;
		int mint_trajectoryIndex;

		// Pyramid info.
		int mint_PyramidTopLevel;
		int mint_PyramidBottomLevel;

		// Output trajectories to file.
		std::vector<Pose*> mPose_trajector;
		std::ofstream mof_OutputFile;

		// DVO weight object.
		WeightCalculation weight_calculation_;
		cv::Mat m_DVOweights;
		InfluenceFunctions::enum_t InfluenceFuntionType;
		float InfluenceFunctionParam;
		ScaleEstimators::enum_t ScaleEstimatorType;
		float ScaleEstimatorParam;
        
		// EdgeDirectVO Eigen image vectors and residual vectors.
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im1;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im2;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im1Final;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_im2Final;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_residual;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_rsquared;

		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_weights;
		Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> m_Jacobian;

        // Depth of reference frame
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_Z;
        // Warped x,y image coordinates
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedX;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedY;

        // Vector of 3D points and Transformed 3D points
        std::vector<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> m_X3DVector;
        Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> m_X3D;

        Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> m_newX3D;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_XFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_YFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_ZFinal;

        // Vectors of Image Gradients
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gx;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gy;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gxFinal;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_gyFinal;
        
        // Mask for edge pixels as well as to prevent out of bounds, NaN, etc.
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::RowMajor> mEM_finalPointMask;
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::RowMajor> mEM_pointMask;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> m_warpedZ;
};

}
#endif //NLSOLVER_H