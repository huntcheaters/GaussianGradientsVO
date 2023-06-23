#include <nlsolver.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdio.h> 
#include <stdlib.h> 
#include <random>
#include <iterator>
#include <algorithm>

#include <configuration.h>


namespace GGMVO{
	using namespace cv;

	// Overload constructors.
	// Note there are 3 indices, two of both reference and current frames and one of trajectory which are initialized to 0, 1 and -1, respectively.
	NLSolver::NLSolver(const int & pyramidLevel) : mint_PyramidBottomLevel(Configuration::PYRAMID_BOTTOMLEVEL), mint_PyramidTopLevel(Configuration::PYRAMID_TOPLEVEL), mint_refFrameIndex(0), mint_curFrameIndex(1), mint_trajectoryIndex(mint_refFrameIndex - 1)
	{
		// Frames
		mOF_refFrame = new OneFrame(Configuration::str_DatasetPath + Configuration::vecstr_RGBfiles[mint_refFrameIndex], Configuration::str_DatasetPath + Configuration::vecstr_Depthfiles[mint_refFrameIndex], mint_refFrameIndex);
		mOF_curFrame = new OneFrame(Configuration::str_DatasetPath + Configuration::vecstr_RGBfiles[mint_curFrameIndex], Configuration::str_DatasetPath + Configuration::vecstr_Depthfiles[mint_curFrameIndex], mint_curFrameIndex);

		int lengthBottomLevel = mOF_curFrame->getWidthPerLevel(mint_PyramidBottomLevel) * mOF_curFrame->getHeightPerLevel(mint_PyramidBottomLevel);

		m_X3DVector.resize(pyramidLevel); // Vector for each pyramid level
		for (size_t i = 0; i < m_X3DVector.size(); ++i)
			m_X3DVector[i].resize(lengthBottomLevel / std::pow(4, i), Eigen::NoChange); //3 Vector for each pyramid for each image pixel

		m_X3D.resize(lengthBottomLevel, Eigen::NoChange);
		m_warpedX.resize(lengthBottomLevel);
		m_warpedY.resize(lengthBottomLevel);
		m_warpedZ.resize(lengthBottomLevel);
		m_gx.resize(lengthBottomLevel);
		m_gxFinal.resize(lengthBottomLevel);
		m_gy.resize(lengthBottomLevel);
		m_gyFinal.resize(lengthBottomLevel);
		m_im1.resize(lengthBottomLevel);
		m_im1Final.resize(lengthBottomLevel);
		m_im2Final.resize(lengthBottomLevel);
		m_ZFinal.resize(lengthBottomLevel);
		m_Z.resize(lengthBottomLevel);

		mEM_pointMask.resize(lengthBottomLevel);

		mof_OutputFile.open(Configuration::str_TrajectoryFile);
	} // end func


	NLSolver::~NLSolver()
	{
		mof_OutputFile.close();
	}

	// VO via SolverGN or SolverLM.
	void NLSolver::VOviaRGBD()
	{
		// Configure reference frame.
		mOF_refFrame->buildAllPyrmaid();

		// Transform 2D pixel coordinates of reference frame into 3D points.
		calculateRef3D();

		// Camera pose
		Pose oPoseCamera;
		oPoseCamera.setIdentityPose();

		Pose oPKFramePose = oPoseCamera;

		// Relative pose between frames intiialized as identity matrix.
		Pose oPRelativePose;

		// Output the beginning point
		std::string firstTimeStamp = Configuration::vecstr_timestamps[0];
		writePose(oPoseCamera, firstTimeStamp); // Output pose of the starting point.

		addPose(oPoseCamera); // Adding camera pose to trajectory.

		// Tracking starts here.
		tinterval t0 = tic_t(); // T interval.
		for (size_t n = 0; mint_curFrameIndex<Configuration::vecstr_RGBfiles.size(); ++n)
		{
			std::cout << std::endl << oPoseCamera << std::endl;

			// Configure current frame.
			mOF_curFrame->buildAllPyrmaid();

			if (n % Configuration::KEYFRAME_INTERVAL == 0)
			{
				oPKFramePose = oPoseCamera;
				oPRelativePose.setIdentityPose();
			}

			//Constant motion assumption
			oPRelativePose.updateKeyFramePose(oPRelativePose.getPoseMatrix(), getLastRelativePose());
			oPRelativePose.setPose(se3ExpEigen(se3LogEigen(oPRelativePose.getPoseMatrix())));

			// Optimization begins with the smallest layer.
			for (int level = mint_PyramidTopLevel; level >= mint_PyramidBottomLevel; --level)
			{
				const Mat cameraMatrix(Configuration::MatK[level]);
				prepareVectors(level);

				float lambda = 0.f;
				float error_last = INF_F;
				float error = error_last;
				for (int i = 0; i < MAX_ITERATIONS_PER_PYRAMID[level]; ++i)
				{
					error_last = error;

					// Guarantee enough points available.
#ifdef TDISTRIBUTION
					error = warpAndProject(oPRelativePose.inversePoseEigen(), level, i);
					if (!isfinite(error))   // Check validity of error result.
					{
						error = warpAndProject(oPRelativePose.inversePoseEigen(), level);
						if (!isfinite(error))
							continue;
					}
#else
					// The original waprAndProject
					error = warpAndProject(oPRelativePose.inversePoseEigen(), level);
#endif

					// Solving optimization.
					if (error < error_last)
					{
						// Update relative pose
						Eigen::Matrix<double, 6, Eigen::RowMajor> del;

						if (Configuration::SOLVER_METHOD == GN)
							SolverGN(level, del);
						else if (Configuration::SOLVER_METHOD == LM)
							SolverLM(level, del);
						//std::cout << del << std::endl;

						if ((del.segment<3>(0)).dot(del.segment<3>(0)) < MIN_TRANSLATION_UPDATE &
							(del.segment<3>(3)).dot(del.segment<3>(3)) < MIN_ROTATION_UPDATE)
							break;

						cv::Mat delMat = se3ExpEigen(del);
						oPRelativePose.updatePose(delMat);
					}
					else
					{
						;
					}
				}
			} 
			oPoseCamera.updateKeyFramePose(oPKFramePose.getPoseMatrix(), oPRelativePose.getPoseMatrix()); // Update camera pose.

			writePose(oPoseCamera, Configuration::vecstr_timestamps[mint_curFrameIndex]); // Write camera pose to file.

			addPose(oPoseCamera); // Add new camera pose to trajectory.

			// Don't time past this part (reading from disk)
			advanceFrameStream();
		} // end-for

		// Time consumption given here.
		double t1 = toc_t(t0);

		std::cout << "- Frames: " << Configuration::vecstr_RGBfiles.size() << std::endl;
		std::cout << "- Time elapsed: " << t1 << "ms" << std::endl;
		return;
	}

	void NLSolver::prepareVectors(int level)
	{
		// Depth of the reference.
		cv2eigen(mOF_refFrame->getDepth(level), m_Z);

		// Mask of the current frame.
		if (Configuration::METRIC_LABEL == GGM || Configuration::METRIC_LABEL == LOG)
			cv2eigen(mOF_curFrame->getMaxMagnitudeInBlk(level), mEM_pointMask);
		else
		{
			std::cerr << "- Unknown feature metric!" << std::endl;
			exit(0);
		}

		cv2eigen(mOF_refFrame->getImageVector(level), m_im1);
		cv2eigen(mOF_curFrame->getImageVector(level), m_im2);
		cv2eigen(mOF_curFrame->getGradientX(level), m_gx);
		cv2eigen(mOF_curFrame->getGradientY(level), m_gy);

		size_t numElements; // Current point number.

		mEM_pointMask = (m_Z.array() <= 0.f).select(0, mEM_pointMask);   // Check depth validity.

		numElements = (mEM_pointMask.array() != 0).count(); //

		m_im1Final.resize(numElements);
		m_XFinal.resize(numElements);
		m_YFinal.resize(numElements);
		m_ZFinal.resize(numElements);
		m_X3D.resize(numElements, Eigen::NoChange);
		mEM_finalPointMask.resize(numElements);
		size_t idx = 0;
		for (int i = 0; i < mEM_pointMask.rows(); ++i)
		{
			if (mEM_pointMask[i] != 0)
			{
				m_im1Final[idx] = m_im1[i];
				m_ZFinal[idx] = m_Z[i];
				m_X3D.row(idx) = (m_X3DVector[level].row(i)).array() * m_Z[i];
				mEM_finalPointMask[idx] = mEM_pointMask[i];
				++idx;
			}
		}

		////////////////////////////////////////////////////////////
		m_Z.resize(numElements);
		m_Z = m_ZFinal;
		mEM_pointMask.resize(numElements);
		mEM_pointMask = mEM_finalPointMask;
	}

	void NLSolver::make3DPoints(const cv::Mat& cameraMatrix, int level)
	{
		m_X3D = m_X3DVector[level].array() * m_Z.replicate(1, m_X3DVector[level].cols()).array();
	}

	// Overlapped function.
	float NLSolver::warpAndProject(const Eigen::Matrix<double, 4, 4>& invPose, int level, const int & iterOnlevel)
	{
		Eigen::Matrix<float, 3, 3> R = (invPose.block<3, 3>(0, 0)).cast<float>();
		Eigen::Matrix<float, 3, 1> t = (invPose.block<3, 1>(0, 3)).cast<float>();

		m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
		m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows());

		const Mat cameraMatrix(Configuration::MatK[level]);
		const float fx = cameraMatrix.at<float>(0, 0);
		const float cx = cameraMatrix.at<float>(0, 2);
		const float fy = cameraMatrix.at<float>(1, 1);
		const float cy = cameraMatrix.at<float>(1, 2);

		const int w = mOF_curFrame->getWidthPerLevel(level);
		const int h = mOF_curFrame->getHeightPerLevel(level);

		m_warpedX.resize(m_X3D.rows());
		m_warpedY.resize(m_X3D.rows());

		m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array()) + cx;
		m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array()) + cy;

		mEM_finalPointMask = mEM_pointMask;

		mEM_finalPointMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, mEM_finalPointMask);


		mEM_finalPointMask = (m_X3D.col(2).array() <= 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = ((m_X3D.col(2).array()).isFinite()).select(mEM_finalPointMask, 0);
		mEM_finalPointMask = ((m_newX3D.row(2).transpose().array()).isFinite()).select(mEM_finalPointMask, 0);

		// Check new projected x coordinates are: 0 <= x < w-1
		mEM_finalPointMask = (m_warpedX.array() < 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedX.array() >= w - 2).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedX.array().isFinite()).select(mEM_finalPointMask, 0);
		// Check new projected x coordinates are: 0 <= y < h-1
		mEM_finalPointMask = (m_warpedY.array() >= h - 2).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedY.array() < 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedY.array().isFinite()).select(mEM_finalPointMask, 0);



		size_t numElements = (mEM_finalPointMask.array() != 0).count();

		if (numElements == 0)                   // Added on 19/12/2022.
			return INF_F;

		m_gxFinal.resize(numElements);
		m_gyFinal.resize(numElements);
		m_im1.resize(numElements);
		m_im2Final.resize(numElements);
		m_XFinal.resize(numElements);
		m_YFinal.resize(numElements);
		m_ZFinal.resize(numElements);

		size_t idx = 0;
		for (int i = 0; i < mEM_finalPointMask.rows(); ++i)
		{
			if (mEM_finalPointMask[i] != 0)
			{
				m_gxFinal[idx] = interpolateVector(m_gx, m_warpedX[i], m_warpedY[i], w);
				m_gyFinal[idx] = interpolateVector(m_gy, m_warpedX[i], m_warpedY[i], w);
				m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
				m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
				m_XFinal[idx] = m_newX3D(0, i);
				m_YFinal[idx] = m_newX3D(1, i);
				m_ZFinal[idx] = m_newX3D(2, i);

				++idx;
			}
		}

		// Resize in terms of final mask size.
		m_residual.resize(numElements);
		m_rsquared.resize(numElements);
		m_weights.resize(numElements);

		m_residual = (m_im1.array() - m_im2Final.array());

		// T-Distribution.
#ifdef TDISTRIBUTION
		cv::Mat residuals;
		cv::eigen2cv(m_residual, residuals);
		computeScale(residuals, level, iterOnlevel);
		weight_calculation_.calculateWeights(residuals, m_DVOweights);
		cv::cv2eigen(m_DVOweights, m_weights);
		m_rsquared = m_residual.array() * m_residual.array();
#else
		m_rsquared = m_residual.array() * m_residual.array();

		m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
		m_weights = (((m_residual.array()).abs()) > HUBER_THRESH).select(HUBER_THRESH / (m_residual.array()).abs(), m_weights);
#endif

		return ((m_weights.array() * m_rsquared.array()).sum() / (float)numElements);

	}

	float NLSolver::warpAndProject(const Eigen::Matrix<double, 4, 4>& invPose, int level)
	{
		Eigen::Matrix<float, 3, 3> R = (invPose.block<3, 3>(0, 0)).cast<float>();
		Eigen::Matrix<float, 3, 1> t = (invPose.block<3, 1>(0, 3)).cast<float>();

		m_newX3D.resize(Eigen::NoChange, m_X3D.rows());
		m_newX3D = R * m_X3D.transpose() + t.replicate(1, m_X3D.rows());

		const Mat cameraMatrix(Configuration::MatK[level]);

		const float fx = cameraMatrix.at<float>(0, 0);
		const float cx = cameraMatrix.at<float>(0, 2);
		const float fy = cameraMatrix.at<float>(1, 1);
		const float cy = cameraMatrix.at<float>(1, 2);

		const int w = mOF_curFrame->getWidthPerLevel(level);
		const int h = mOF_curFrame->getHeightPerLevel(level);

		m_warpedX.resize(m_X3D.rows());
		m_warpedY.resize(m_X3D.rows());

		m_warpedX = (fx * (m_newX3D.row(0)).array() / (m_newX3D.row(2)).array()) + cx;
		m_warpedY = (fy * (m_newX3D.row(1)).array() / (m_newX3D.row(2)).array()) + cy;

		mEM_finalPointMask = mEM_pointMask;

		mEM_finalPointMask = (m_newX3D.row(2).transpose().array() <= 0.f).select(0, mEM_finalPointMask);

		mEM_finalPointMask = (m_X3D.col(2).array() <= 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = ((m_X3D.col(2).array()).isFinite()).select(mEM_finalPointMask, 0);
		mEM_finalPointMask = ((m_newX3D.row(2).transpose().array()).isFinite()).select(mEM_finalPointMask, 0);

		// Check new projected x coordinates are: 0 <= x < w-1
		mEM_finalPointMask = (m_warpedX.array() < 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedX.array() >= w - 2).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedX.array().isFinite()).select(mEM_finalPointMask, 0);
		// Check new projected x coordinates are: 0 <= y < h-1
		mEM_finalPointMask = (m_warpedY.array() >= h - 2).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedY.array() < 0.f).select(0, mEM_finalPointMask);
		mEM_finalPointMask = (m_warpedY.array().isFinite()).select(mEM_finalPointMask, 0);

		size_t numElements = (mEM_finalPointMask.array() != 0).count();
		m_gxFinal.resize(numElements);
		m_gyFinal.resize(numElements);
		m_im1.resize(numElements);
		m_im2Final.resize(numElements);
		m_XFinal.resize(numElements);
		m_YFinal.resize(numElements);
		m_ZFinal.resize(numElements);

		size_t idx = 0;
		for (int i = 0; i < mEM_finalPointMask.rows(); ++i)
		{
			if (mEM_finalPointMask[i] != 0)
			{
				m_gxFinal[idx] = interpolateVector(m_gx, m_warpedX[i], m_warpedY[i], w);
				m_gyFinal[idx] = interpolateVector(m_gy, m_warpedX[i], m_warpedY[i], w);
				m_im1[idx] = m_im1Final[i];//interpolateVector(m_im1, m_warpedX[i], m_warpedY[i], w);
				m_im2Final[idx] = interpolateVector(m_im2, m_warpedX[i], m_warpedY[i], w);
				m_XFinal[idx] = m_newX3D(0, i);
				m_YFinal[idx] = m_newX3D(1, i);
				m_ZFinal[idx] = m_newX3D(2, i);

				++idx;
			}
		}


		// Resize in terms of final point mask size.
		m_residual.resize(numElements);
		m_rsquared.resize(numElements);
		m_weights.resize(numElements);

		m_residual = (m_im1.array() - m_im2Final.array());
		m_rsquared = m_residual.array() * m_residual.array();

		// Using Huber weighting.
		m_weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>::Ones(numElements);
		m_weights = (((m_residual.array()).abs()) > HUBER_THRESH).select(HUBER_THRESH / (m_residual.array()).abs(), m_weights);

		return ((m_weights.array() * m_rsquared.array()).sum() / (float)numElements);

	}

	// Interpolation.
	float NLSolver::interpolateVector(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor>& toInterp, float x, float y, int w) const
	{
		int xi = (int)x;
		int yi = (int)y;
		float dx = x - xi;
		float dy = y - yi;
		float dxdy = dx * dy;
		int topLeft = w * yi + xi;
		int topRight = topLeft + 1;
		int bottomLeft = topLeft + w;
		int bottomRight = bottomLeft + 1;

		//               x                x+1
		//       ======================================
		//  y    |    topLeft      |    topRight      |
		//       ======================================
		//  y+w  |    bottomLeft   |    bottomRight   |
		//       ======================================
		return  dxdy * toInterp[bottomRight]
			+ (dy - dxdy) * toInterp[bottomLeft]
			+ (dx - dxdy) * toInterp[topRight]
			+ (1.f - dx - dy + dxdy) * toInterp[topLeft];
	}

	// 3D of the reference image.
	void NLSolver::calculateRef3D()
	{

		for (int level = 0; level < Configuration::PyramidLevel; ++level)
		{
			const cv::Mat cameraMatrix(Configuration::MatK[level]);

			int w = mOF_refFrame->getWidthPerLevel(level);
			int h = mOF_refFrame->getHeightPerLevel(level);

			const float fx = cameraMatrix.at<float>(0, 0);
			const float cx = cameraMatrix.at<float>(0, 2);
			const float fy = cameraMatrix.at<float>(1, 1);
			const float cy = cameraMatrix.at<float>(1, 2);
			const float fxInv = 1.f / fx;
			const float fyInv = 1.f / fy;

			for (int y = 0; y < h; ++y)
			{
				for (int x = 0; x < w; ++x)
				{
					int idx = y * w + x;
					m_X3DVector[level].row(idx) << (x - cx) * fxInv, (y - cy) * fyInv, 1.f;
				} // end 2-nd for
			} // end 1-st for
		} // end 0-th for
	} // end func

	inline
		bool NLSolver::checkBounds(float x, float xlim, float y, float ylim, float oldZ, float newZ, bool edgePixel)
	{
		return ((edgePixel)& (x >= 0) & x < xlim & y >= 0 & y < ylim & oldZ >= 0. & newZ >= 0.);

	}

	// Output pose to file.
	void NLSolver::writePose(const Pose& pose, const std::string & timestamp)
	{
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T;
		cv::Mat pmat = pose.getPoseMatrix();
		cv::cv2eigen(pmat, T);
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = T.block<3, 3>(0, 0);
		Eigen::Matrix<double, 3, Eigen::RowMajor> t = T.block<3, 1>(0, 3);
		Eigen::Quaternion<double> quat(R);

		mof_OutputFile << std::fixed << std::showpoint << timestamp;
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << t[0];
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << t[1];
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << t[2];
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << quat.x();
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << quat.y();
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << quat.z();
		mof_OutputFile << " ";
		mof_OutputFile << std::setprecision(OUTPUT_PRECISION) << std::fixed << std::showpoint << quat.w();
		mof_OutputFile << std::endl;
	}

	// NLSolver Trajectory relative pose
	const cv::Mat NLSolver::getLastRelativePose() const
	{
		if (mint_trajectoryIndex > 1)
			return (mPose_trajector[mint_trajectoryIndex - 1]->inversePose() * mPose_trajector[mint_trajectoryIndex]->getPoseMatrix());
		else
			return cv::Mat::eye(4, 4, CV_64FC1);
	}

	// NLSolver trajecotry current pose
	const Pose& NLSolver::getCurrentPose() const
	{
		return *(mPose_trajector.back());
	}

	// NLSolver Trajectory add new pose
	void NLSolver::addPose(Pose& newPose)
	{
		mint_trajectoryIndex++; // Pose index starts from 0.
		mPose_trajector.push_back(new Pose(newPose));
	}

	// NLSolver advance frames
	void NLSolver::advanceFrameStream()
	{
		++mint_curFrameIndex;
		++mint_refFrameIndex;
		if (mint_refFrameIndex % Configuration::KEYFRAME_INTERVAL == 0)
		{
			delete mOF_refFrame;
			mOF_refFrame = mOF_curFrame;
		}
		else
		{
			delete mOF_curFrame;
		}

		mOF_curFrame = new OneFrame(Configuration::str_DatasetPath + Configuration::vecstr_RGBfiles[mint_curFrameIndex], Configuration::str_DatasetPath + Configuration::vecstr_Depthfiles[mint_curFrameIndex], mint_curFrameIndex);
	}

	// DVO Student T-distribution configuration. 
	void NLSolver::configSTDweight(const ScaleEstimators::enum_t & scaleType)
	{
		InfluenceFuntionType = InfluenceFunctions::TDistribution;
		InfluenceFunctionParam = TDistributionInfluenceFunction::DEFAULT_DOF;
		ScaleEstimatorType = scaleType;

		ScaleEstimatorParam = TDistributionScaleEstimator::DEFAULT_DOF;

		weight_calculation_.scaleEstimator(ScaleEstimators::get(ScaleEstimatorType))
			.scaleEstimator()->configure(ScaleEstimatorParam);
		weight_calculation_.influenceFunction(InfluenceFunctions::get(InfluenceFuntionType))
			.influenceFunction()->configure(InfluenceFunctionParam);
	}


	// inline function for calculating scale. 
	inline void NLSolver::computeScale(const cv::Mat& residuals, const int & currentLevel, const int & currentIterationOnLevel)
	{
		if (currentIterationOnLevel == FIRST_ITERATION_ONLEVEL)
		{
			weight_calculation_.calculateScale(residuals);
		}

		// TODO GA or Jf.
	}

	// Solving with Gaussian-Newton
	void NLSolver::SolverGN(const int level, Eigen::Matrix<double, 6, Eigen::RowMajor>& poseupdate)
	{
		const cv::Mat cameraMatrix(Configuration::MatK[level]);
		const float fx = cameraMatrix.at<float>(0, 0);
		const float cx = cameraMatrix.at<float>(0, 2);
		const float fy = cameraMatrix.at<float>(1, 1);
		const float cy = cameraMatrix.at<float>(1, 2);

		size_t numElements = m_im2Final.rows();
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::RowMajor> Z2 = m_ZFinal.array() * m_ZFinal.array();

		m_Jacobian.resize(numElements, Eigen::NoChange);
		m_Jacobian.col(0) = m_weights.array() * fx * (m_gxFinal.array() / m_ZFinal.array());

		m_Jacobian.col(1) = m_weights.array() * fy * (m_gyFinal.array() / m_ZFinal.array());

		m_Jacobian.col(2) = -m_weights.array()* (fx * (m_XFinal.array() * m_gxFinal.array()) + fy * (m_YFinal.array() * m_gyFinal.array()))
			/ (Z2.array());

		m_Jacobian.col(3) = -m_weights.array() * (fx * m_XFinal.array() * m_YFinal.array() * m_gxFinal.array() / Z2.array()
			+ fy *(1.f + (m_YFinal.array() * m_YFinal.array() / Z2.array())) * m_gyFinal.array());

		m_Jacobian.col(4) = m_weights.array() * (fx * (1.f + (m_XFinal.array() * m_XFinal.array() / Z2.array())) * m_gxFinal.array()
			+ fy * (m_XFinal.array() * m_YFinal.array() * m_gyFinal.array()) / Z2.array());

		m_Jacobian.col(5) = m_weights.array() * (-fx * (m_YFinal.array() * m_gxFinal.array()) + fy * (m_XFinal.array() * m_gyFinal.array()))
			/ m_ZFinal.array();

		m_residual.array() *= m_weights.array();

		poseupdate = -((m_Jacobian.transpose() * m_Jacobian).cast<double>()).ldlt().solve((m_Jacobian.transpose() * m_residual).cast<double>());
	}


	// Solving with Levenberg-Marquardt. Not given in this version.
	void NLSolver::SolverLM(const int level, Eigen::Matrix<double, 6, Eigen::RowMajor>& poseupdate)
	{
		// H = J^T*J, g = -J^T*e
		
		// Calculate u.

		// poseupdate;

		// Taylor similarity;

		// Update;
		;
 	} // end SolverLM

} //end namespace NLSolver


