#include <configuration.h>

namespace GGMVO{

	// Static members initialization
	cv::FileStorage Configuration::cvFSConfig;
	int Configuration::PyramidLevel;                // Pyramid level number.
	std::vector<cv::Mat> Configuration::MatK;       // Camera intrinsic in float.

	bool Configuration::bUsingPyramid;		
	int Configuration::SOLVER_METHOD;				// 1 GN, 2 LM
	int Configuration::SCALE_ESTIMATOR;             // 1 TD, 2 MAD
	int Configuration::METRIC_LABEL;				// 1 GGM, 2 LOG
	int Configuration::KEYFRAME_INTERVAL;			// 1 means frame-by-frame tracking

	int Configuration::PYRAMID_TOPLEVEL;
	int Configuration::PYRAMID_BOTTOMLEVEL;

	const double Configuration::PIXEL_TO_METER_SCALE(0.0002); // 0.0002f = 1.f/5000.f

	std::string Configuration::str_DatasetPath;
	std::string Configuration::str_TrajectoryFile;
	std::vector<std::string> Configuration::vecstr_RGBfiles;
	std::vector<std::string> Configuration::vecstr_Depthfiles;
	std::vector<std::string> Configuration::vecstr_timestamps;

	// Configuration function
	void Configure(const std::string & configAbPath)
	{
		std::cout << "Direct RGB-D Visual Odometry via Gaussian Gradient metrics." << std::endl <<
			"First uploaded to IEEE TechRxiv by YAO Zhigang in 2023. " << std::endl;

		std::cout << std::endl << "- Have you already checked your configuration file before running?" << std::endl;
		std::cout << "  (y/n)" ;
		char answer;
		std::cin >> answer;
		if (answer == 'y' || answer == 'Y')
			std::cout << "- Configuring ..." << std::endl;
		else
			exit(0);

		assert(~configAbPath.empty());  // empty() returns 1 if it is empty and 0 for otherwise.
		Configuration::cvFSConfig = cv::FileStorage(configAbPath, cv::FileStorage::READ); // Read config file.

		// Read pyramid setting.
		if ((int)Configuration::cvFSConfig["UsingPyramid"])
		{
			Configuration::bUsingPyramid = true;
			Configuration::PyramidLevel = Configuration::cvFSConfig["PyramidLevel"];
			std::cout << "  Using pyramid of " << Configuration::PyramidLevel << " levels." << std::endl;
		}
		else
		{
			Configuration::bUsingPyramid = false;
			Configuration::PyramidLevel = Configuration::cvFSConfig["PyramidLevel"];
			std::cout << "  Pyramid not used" << std::endl;
		}

		Configuration::PYRAMID_TOPLEVEL = Configuration::PyramidLevel - 1;
		Configuration::PYRAMID_BOTTOMLEVEL = 0;

		// Scale estimator
		Configuration::SOLVER_METHOD = Configuration::cvFSConfig["SOLVER_METHOD"];
		Configuration::METRIC_LABEL = Configuration::cvFSConfig["METRIC_LABEL"];
		Configuration::SCALE_ESTIMATOR = Configuration::cvFSConfig["SCALE_ESTIMATOR"];
		Configuration::KEYFRAME_INTERVAL = Configuration::cvFSConfig["KEYFRAME_INTERVAL"];

		// Camera intrinsic
		float fx = Configuration::cvFSConfig["Camera.fx"];
		float fy = Configuration::cvFSConfig["Camera.fy"];
		float cx = Configuration::cvFSConfig["Camera.cx"];
		float cy = Configuration::cvFSConfig["Camera.cy"];
		for (int i = 0; i < Configuration::PyramidLevel; i++) // the 0-th level is the original image.
		{
			float denominator(std::pow(2., i));
			float fxi = fx / denominator;
			float fyi = fy / denominator;
			float cxi = cx / denominator;
			float cyi = cy / denominator;
			cv::Mat tmpK = (cv::Mat_<float>(3, 3) << fxi, 0, cxi, 0, fyi, cyi, 0, 0, 1);
			Configuration::MatK.push_back(tmpK);
			std::cout << "  Camera instrinsic matrix of level " << i << " :" << std::endl << "  " << tmpK << std::endl;
		}

		// Dataset absolute path
		Configuration::str_DatasetPath = Configuration::cvFSConfig["DatasetPath"];
		std::cout << "  Dataset: " << Configuration::str_DatasetPath << std::endl;
		LoadImages(Configuration::str_DatasetPath + "association.txt", Configuration::vecstr_RGBfiles, Configuration::vecstr_Depthfiles, Configuration::vecstr_timestamps);
		std::cout << "  " << Configuration::vecstr_RGBfiles.size() << " frames loaded." << std::endl;

		// Output file
		Configuration::str_TrajectoryFile = Configuration::str_DatasetPath + Configuration::cvFSConfig["TrajectoryFile"];
		std::cout << "  Output file: " << Configuration::str_TrajectoryFile << std::endl;
	} // end function


	// ORBSLAM2 LoadImages()
	void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
		std::vector<std::string> &vstrImageFilenamesD, std::vector<std::string> &vTimestamps)
	{
		std::ifstream fAssociation;
		fAssociation.open(strAssociationFilename.c_str());
		while (!fAssociation.eof())
		{
			std::string s;
			getline(fAssociation, s);
			if (!s.empty())
			{
				std::stringstream ss;
				ss << s;
				std::string t;
				std::string sRGB, sD;
				ss >> t;
				vTimestamps.push_back(t);
				ss >> sRGB;
				vstrImageFilenamesRGB.push_back(sRGB);
				ss >> t;
				ss >> sD;
				vstrImageFilenamesD.push_back(sD);
			} // end if
		} // end wile
	} // end function

} // end namespace Configuration
