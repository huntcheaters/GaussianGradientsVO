// Configuration of problem model.
// 09/12/2022.
// zyao@ncut.edu.cn
#pragma once
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// General IO and libraries of C++.
#include <iostream>      
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <io.h>          // For folder traversal.

// OpenCV libraries.
#include <opencv\cv.h>
#include <opencv2\core\core.hpp>
#include <opencv2/opencv.hpp>

namespace GGMVO{

#define TDISTRIBUTION 1
#define WITH_BLUR 1
#define WITH_SHARPEN 1
#define PLAIN 1

	enum ScaleEstimator{
		MAD = 1, 
		TD = 2
	};

	enum Metric{
		GGM = 1, 
		LOG = 2
	};

	enum SOLVER_METHOD{
		GN = 1,
		LM = 2
	};

	// ORB SLAM2 LoadImages().
	void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
		std::vector<std::string> &vstrImageFilenamesD, std::vector<std::string> &vTimestamps);

	// We do it with a configuration file as it is done by ORBSLAM2.
	void Configure(const std::string & configAbPath);

	// Define an independent class with no entity for doing configuration
	class Configuration{

	public:
		static cv::FileStorage cvFSConfig;
		static std::vector<cv::Mat> MatK;       // Camera intrinsic in float.

		static bool bUsingPyramid;
		static int SOLVER_METHOD;				// 1 GN, 2 LM
		static int SCALE_ESTIMATOR;             // 1 TD, 2 MAD
		static int METRIC_LABEL;				// 1 GGM, 2 LOG
		static int KEYFRAME_INTERVAL;			// 1 means frame-by-frame.

		static int PyramidLevel;                // Pyramid level number.
		static int PYRAMID_TOPLEVEL;
		static int PYRAMID_BOTTOMLEVEL;

		static const double PIXEL_TO_METER_SCALE; // 0.0002f = 1.f/5000.f

		static std::string str_DatasetPath;
		static std::string str_TrajectoryFile;
		static std::vector<std::string> vecstr_RGBfiles;
		static std::vector<std::string> vecstr_Depthfiles;
		static std::vector<std::string> vecstr_timestamps;
	}; // end class

} // end namespace

#endif // CONFIGURATION_