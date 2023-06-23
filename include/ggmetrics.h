// Gaussian gradient metrics.
// Things obtained are the same as the original Matlab code. This guarantees the correctness.
// 09/10/2022.
// zyao@ncut.edu.cn
#pragma once
#ifndef GGMETRICS_H
#define GGMETRICS_H

#include <configuration.h>

namespace GGMVO{

	// CV_32FC1 to CV_8U into [0,255]
	void cv32FC1to8U(const cv::Mat & src, cv::Mat & dst);

	// Simply encapsulate as an independent class.
	class GGMetrics{
		// constructor & destructor
	public:
		GGMetrics();
		~GGMetrics();

	public:
		// Implementations for metrics of BIQA paper.
		cv::Mat fspecialGaussian(const float & sigma, const int & size);
		cv::Mat fspecialLOG(const float & sigma, const int & size);
		void generateKernelGaussian(cv::Mat & winx, cv::Mat & winy, const cv::Mat & window1, const float & sigma = 0.5, const int & size= 7);
		// We give GM and LOG via separate functions for flexibility.
		void gradientMagnitude(const cv::Mat & src, const cv::Mat & srcColor, cv::Mat & dstMagnitude, cv::Mat & dstPhase, cv::Mat & dstGrad, const float & sigma = 0.5, const int & length = 7);
		void biqaLOG(const cv::Mat & src, cv::Mat & dstf, const float & sigma = 0.5, const int & size = 5);

		// The joint normalization is not employed in experiments.
		void jointnormalize(const cv::Mat & grad_im, const cv::Mat & log_im, cv::Mat & jan_grad_im, cv::Mat & jan_log_im, const float & sigma);

	private:
		cv::Mat mMat_GGM;
		cv::Mat mMat_LOG;
	};
}

#endif 