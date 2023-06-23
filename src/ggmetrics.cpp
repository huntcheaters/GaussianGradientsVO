// The Gaussian gradients metrics via low order gaussian derivatives.
// This C++ implementation is only for experiments of my VO paper 
// https://www.techrxiv.org/articles/preprint/RGB-D_Visual_Odometry_via_Low_Order_Gaussian_Gradient_Metrics/22132544.
// No guarantees or warranties of any kind are expressed or implied 
// for any other uses of the code.
// The BIQA paper is available at
// https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6894197.
// For implementation, we recommend readers reading their Matlab code.

// Note that one can check the result via visualization or text
// outputs in comparison with their counterparts in Matlab.

#include <ggmetrics.h>

namespace GGMVO{

	// CV_32FC1 to CV_8U into [0,255]
	// https://stackoverflow.com/questions/14539498/change-type-of-mat-object-from-cv-32f-to-cv-8u
	void cv32FC1to8U(const cv::Mat & src, cv::Mat & dst)
	{
		cv::Mat tMat = src.clone();
		double Min, Max;
		cv::Point MinLoc, MaxLoc;
		cv::minMaxLoc(tMat, &Min, &Max, &MinLoc, &MaxLoc);
		if (Min != Max){
			tMat -= Min;
			tMat.convertTo(dst, CV_8U, 255.0 / (Max - Min));
		}
	}

	// ----------------------------------------------------------------------------------------------//
	// Default constructor and destructor.
	GGMetrics::GGMetrics()
	{

	}

	GGMetrics::~GGMetrics()
	{

	}

	// Kernel function refers to the paper.

	// To generate a gaussian window similar to fspecial() in Matlab.
	// https://blog.csdn.net/zhengxinjie2/article/details/84940696
	// Parameters in default: const float & sigma = 0.5, const int & size = 7
	cv::Mat GGMetrics::fspecialGaussian(const float & sigma, const int & size)
	{
		int M = 0; // Gaussian window size.
		if (size != 2 * ceil(3 * sigma) + 1 + 2)
			M = size;
		else
			M = 2 * ceil(3 * sigma) + 1 + 2; // window1 size. It is a 7-by-7 window in default, i.e. 2 times of sigma (0.5).

		cv::Mat K = cv::Mat::zeros(M, M, CV_32FC1);  // To save window1.

		int m = M / 2;
		int n = M / 2;
		for (int i = (-m); i <= m; i++)
		{
			int row = i + m;
			for (int j = (-n); j <= n; j++)
			{
				int col = j + n;
				float v = exp(-(1.0*i*i + 1.0*j*j) / (2 * pow(sigma, 2))); // Note that there is no 2PI in the denominator, i.e. same to Matlab.
				K.at<float>(row, col) = v;
			}
		}
		cv::Scalar p;
		p = cv::sum(K);
		K.convertTo(K, CV_32FC1, 1 / p.val[0]);   // Normalize.

		return K;
	}

	// fspecial for LOG
	cv::Mat GGMetrics::fspecialLOG(const float & sigma, const int & size)
	{
		// NOT GIVEN IN THIS VERSION.
		return cv::Mat();
	}

	// Generate kernel X and kernel Y.
	void GGMetrics::generateKernelGaussian(cv::Mat & winx, cv::Mat & winy, const cv::Mat & window1, const float & sigma, const int & size)
	{
		int M = 0;   // Original windows size.
		if (size != 2 * ceil(3 * sigma) + 1 + 2)
			M = size;
		else
			M = 2 * ceil(3 * sigma) + 1 + 2;

		int M2 = M - 2; // size of winx and winy.
		winx = cv::Mat::zeros(M2, M2, CV_32FC1);  // Allocating memory for winx and winy.
		winy = winx.clone();

		// Doing differential for derivative.
		double sumx = 0, sumy = 0;

		// Calculate winx and winy.
		for (int i = 1; i < (M - 1); i++)
		{
			std::cout << std::endl;
			for (int j = 1; j < (M - 1); j++)
			{
				//printf("%f\t", window1.at<float>(i, j));
				winx.at<float>(i - 1, j - 1) = window1.at<float>(i, j) - window1.at<float>(i, j + 1);
				winy.at<float>(i - 1, j - 1) = window1.at<float>(i, j) - window1.at<float>(i + 1, j);
				sumx += abs(winx.at<float>(i - 1, j - 1));
				sumy += abs(winy.at<float>(i - 1, j - 1));
			}
		}

		// Normalize and output winx or winy for debug.
		for (int i = 0; i < M2; i++)
		{
			std::cout << std::endl;
			for (int j = 0; j < M2; j++)
			{
				winx.at<float>(i, j) /= sumx;
				//printf("%f\t", winx.at<float>(i , j ));
				winy.at<float>(i, j) /= sumy;
			}
		}
		//std::cin.ignore();
	}


	// Calculate gradient magnitude with Gaussian kernel.
	void GGMetrics::gradientMagnitude(const cv::Mat & src, const cv::Mat & srcColor, cv::Mat & dstMagnitude, cv::Mat & dstPhase, cv::Mat & dstGrad, const float & sigma, const int & length)
	{
		cv::Mat kernelX, kernelY;

		generateKernelGaussian(kernelX, kernelY, fspecialGaussian(sigma, length), sigma, length);  // Checked with output.

		cv::Mat gx, gy, gx2, gy2; // To save convolution result.
		cv::Mat srcf;

		// Convert into float first. Then we can get gx and gy in float.
		src.convertTo(srcf, CV_32FC1);

		cv::Point anchor = cv::Point(-1, -1);

		cv::filter2D(srcf, gx, -1, kernelX, anchor, 0, cv::BORDER_CONSTANT);
		cv::filter2D(srcf, gy, -1, kernelY, anchor, 0, cv::BORDER_CONSTANT);

		// Display filtered images.
		//cv::Mat tgx, tgy, tgm;
		//cv::normalize(gx, tgx, 0, 1, cv::NORM_MINMAX);
		//cv::imshow("convX", gx);
		//cv::waitKey(0);
		//cv::normalize(gx, tgx, 0, 1, cv::NORM_MINMAX);
		//cv::imshow("convY", gy);
		//cv::waitKey(0);

		cv::pow(gx, 2, gx2);
		cv::pow(gy, 2, gy2);
		cv::Mat squared_gm;
		cv::add(gx2, gy2, squared_gm);
		cv::sqrt(squared_gm, dstMagnitude);  // dst is still a float and note that sqrt requires a float input.

		cv::phase(gx, gy, dstPhase, true); // Phase
		cv::addWeighted(gx, 0.5, gy, 05, 0, dstGrad); // Grad
	}

	// BIQA LOG.
	void GGMetrics::biqaLOG(const cv::Mat & src, cv::Mat & dstf, const float & sigma, const int & size)
	{
		// NOT GIVEN IN THIS VERSION.
		;
	}

	// Joint adaptive normalization.
	void GGMetrics::jointnormalize(const cv::Mat & grad_im, const cv::Mat & log_im, cv::Mat & jan_grad_im, cv::Mat & jan_log_im, const float & sigma)
	{
		//%Normalization
		// NOT GIVEN IN THIS VERSION
	}
}