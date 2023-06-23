#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Eigen/Dense"
#include <sstream>
#include <fstream>

#include <algorithm>
#include <utility>
#include <functional>
#include <cctype>
#include <cstdlib>
#include <string>

#include <tinterval.h>
#include <nlsolver.h>

#define Eigen_Vectorize

using namespace Eigen;
using namespace GGMVO;

int main(int argc, char ** argv)
{

	if (argc != 2)
	{
		std::cerr << std::endl << "Usage: ./GMVO absolute_path_of_the_configuration_file" << std::endl;
		return 1;
	}

	// Configure system
	Configure(argv[1]);

	// VO handler
	NLSolver oNLSolver(Configuration::PyramidLevel);
	if (Configuration::SCALE_ESTIMATOR == GGMVO::MAD)
		oNLSolver.configSTDweight(ScaleEstimators::MAD);
	else if (Configuration::SCALE_ESTIMATOR == GGMVO::TD)
		oNLSolver.configSTDweight(ScaleEstimators::TDistribution);
	else
	{
		std::cerr << "- Unknown scale estimator type!" << std::endl;
		return 1;
	}

	// Motion estimation with tracking RGB-D streams
	oNLSolver.VOviaRGBD();

	// Remainder of running finished.
	std::vector<int> a(2);
	std::cout << a[3] << std::endl;

	return 0;

	// To show edge detection's failure on fr3 datasets.
	/*
	cv::Mat img = imread("F:\\slamData\\rgbd_dataset_freiburg3_nostructure_notexture_far\\rgb\\1341840843.278543.png");
	//std::string source = "../rgbd_dataset_freiburg1_xyz/rgb";
	//std::vector<std::string> files;

	cv::Mat imggray = imread("F:\\slamData\\rgbd_dataset_freiburg3_nostructure_notexture_far\\rgb\\1341840843.310587.png", cv::ImreadModes::IMREAD_GRAYSCALE);
	cv::Mat imgdepth = imread("F:\\slamData\\rgbd_dataset_freiburg3_nostructure_notexture_far\\depth\\1341840843.311514.png", cv::ImreadModes::IMREAD_UNCHANGED);

	cv::Mat img_thresh, img_edge; //not used
	//cv::GaussianBlur( m_pyramidImageUINT[i], m_pyramidImageUINT[i], Size(3,3), EdgeVO::Settings::SIGMA);
	/// Canny detector
	float upperThreshold = cv::threshold(imggray, img_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	float lowerThresh = EdgeVO::Settings::CANNY_RATIO * upperThreshold;
	cv::Canny(imggray, img_edge, lowerThresh, upperThreshold, 3, true);
	cv::imshow("Edge", img_edge);
	cv::waitKey(0);

	cv::imshow("depth ori", imgdepth);
	cv::waitKey(0);

	std::vector<cv::Point> idxes;
	cv::findNonZero(img_edge, idxes);
	std::cout << imgdepth.type() << "|" << imgdepth.channels() << std::endl;
	for (int i = 0; i < idxes.size(); i++)
	{
		std::cout << imgdepth.at<ushort>(idxes[i].y, idxes[i].x) << "  ";
		imgdepth.at<ushort>(idxes[i].y, idxes[i].x) = 65535;
	}

	cv::imshow("depth new", imgdepth);
	cv::waitKey(0); */
}
