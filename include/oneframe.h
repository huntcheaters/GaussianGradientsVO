// A single frame structure.
// 31/08/2022.
// zyao@ncut.edu.cn

#pragma once
#ifndef ONEFRAME_H
#define ONEFRAME_H

#include <configuration.h>
#include <ggmetrics.h>

namespace GGMVO{

	// A single frame
	class OneFrame{
		// constructor & destructor
	public:
		OneFrame();
		~OneFrame();

		// Constructor with absolute paths of image pair synchronized.
		OneFrame(const std::string & rgbAbPath,  const std::string & depthAbPath, const int & frameIndex);

	public:
		void buildImgPyramid(const cv::Mat & src, std::vector<cv::Mat> & dst, const int & pyramidSize, const int & cvInterpolationFlag);
		void buildGGMPyramid();
		void buildAllPyrmaid();
		int corePixelInBlk(const cv::Mat & poMSrcMagImg, const cv::Mat & SrcDepth, cv::Mat & poMDstMaxMagInBlkImg, const cv::Rect2d & poOverlapped2dR, std::vector<cv::Point2d> & pvCVP, const float & global_min, const int & blk_sz = 8);   // 31/08/2022.

		// EdgeDirectVO pixel-wise gradients
		void PixelWiseGradientX(cv::Mat& src, cv::Mat& dst);
		void PixelWiseGradientY(cv::Mat& src, cv::Mat& dst);
		void buildPixelWiseGradientPyramid();
		cv::Mat getMaxMagnitudeInBlk(int level) const; // 31/08/2022.
		cv::Mat getImageVector(const int & level) const;
		cv::Mat getGradientX(const int & level) const;
		cv::Mat getGradientY(const int & level) const;

	public:
		int getHeightPerLevel(const int & level);
		int getWidthPerLevel(const int & level);
		cv::Mat OneFrame::getDepth(int level) const;       // Depth per level

	private:
		// Gaussian gradient metrics' handler
		GGMetrics mGGM_Handler;

	private:
		// Memories for images
		cv::Mat mMat_RGB;
		cv::Mat mMat_Gray;
		cv::Mat mMat_Depth;
		std::vector<cv::Mat> mvecMat_PyramidRGB;
		std::vector<cv::Mat> mvecMat_PyramidGray;
		std::vector<cv::Mat> mvecMat_PyramidGray32FC1;
		std::vector<cv::Mat> mvecMat_PyramidDepth;
		std::string mstr_RGBfname;
		std::string mstr_Depthfname;
		int mint_FrameIndex;

		// Memories for features
		std::vector<cv::Mat> mvecMat_PyramidGrayGGM;
		std::vector<cv::Mat> mvecMat_PyramidGrayGGMPhase;
		std::vector<cv::Mat> mvecMat_PyramidGrayLOG;
		std::vector<cv::Mat> mvecMat_PyramidGrayMaxGGMInBlk;

		// Memories for pixel-wise gradients
		std::vector<cv::Mat> mvecMat_PyramidGrayIdx;
		std::vector<cv::Mat> mvecMat_PyramidGrayIdy;

		bool mb_thresh;
		double md_globalmin;
		
	};
}
#endif 