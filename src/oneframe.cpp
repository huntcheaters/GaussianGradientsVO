// A single frame.

#include <oneframe.h>

namespace GGMVO{

	// Default constructor and destructor.
	OneFrame::OneFrame()
	{

	}

	OneFrame::~OneFrame()
	{
		mvecMat_PyramidRGB.clear();
		mvecMat_PyramidGray.clear();
		mvecMat_PyramidGray32FC1.clear();
		mvecMat_PyramidDepth.clear();
		mvecMat_PyramidGrayGGM.clear();
		mvecMat_PyramidGrayLOG.clear();
		mvecMat_PyramidGrayGGMPhase.clear();
		mvecMat_PyramidGrayMaxGGMInBlk.clear();

		mvecMat_PyramidGrayIdx.clear();
		mvecMat_PyramidGrayIdy.clear();
	}

	// Overload constructor(s)
	OneFrame::OneFrame(const std::string & rgbAbPath, const std::string & depthAbPath, const int & frameIndex) : mMat_RGB(cv::imread(rgbAbPath, cv::IMREAD_UNCHANGED)),
		mMat_Depth(cv::imread(depthAbPath, cv::IMREAD_UNCHANGED)), mMat_Gray(cv::imread(rgbAbPath, cv::IMREAD_GRAYSCALE)), 
		mstr_RGBfname(rgbAbPath), mstr_Depthfname(depthAbPath), mint_FrameIndex(frameIndex)
	{
		if (Configuration::bUsingPyramid)
		{
			mvecMat_PyramidRGB.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGray.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGray32FC1.resize(Configuration::PyramidLevel);
			mvecMat_PyramidDepth.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGrayGGM.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGrayLOG.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGrayGGMPhase.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGrayMaxGGMInBlk.resize(Configuration::PyramidLevel);

			mvecMat_PyramidGrayIdx.resize(Configuration::PyramidLevel);
			mvecMat_PyramidGrayIdy.resize(Configuration::PyramidLevel);
		}

		md_globalmin = 10.0f;  // added on 02/11/2022
		mb_thresh = false;

		mvecMat_PyramidGray[0] = mMat_Gray.clone();
		mvecMat_PyramidGray32FC1[0] = mMat_Gray;
		mvecMat_PyramidGray32FC1[0].convertTo(mvecMat_PyramidGray32FC1[0], CV_32FC1);
		mvecMat_PyramidDepth[0] = mMat_Depth;
		mvecMat_PyramidDepth[0].convertTo(mvecMat_PyramidDepth[0], CV_32FC1, Configuration::PIXEL_TO_METER_SCALE);
	}

	// --------------------------------------------------------------//
	void OneFrame::buildAllPyrmaid()
	{

		buildImgPyramid(mvecMat_PyramidGray32FC1[0], mvecMat_PyramidGray32FC1, Configuration::PyramidLevel, cv::INTER_LINEAR);
		cv::buildPyramid(mvecMat_PyramidGray[0], mvecMat_PyramidGray, Configuration::PyramidLevel - 1);

		buildImgPyramid(mvecMat_PyramidDepth[0], mvecMat_PyramidDepth, Configuration::PyramidLevel, cv::INTER_CUBIC);

		if (Configuration::METRIC_LABEL == GGM || Configuration::METRIC_LABEL == LOG)
			buildGGMPyramid();

		buildPixelWiseGradientPyramid();
		//createPyramid(m_pyramidDepth[0], m_pyramidDepth, EdgeVO::Settings::PYRAMID_DEPTH, cv::INTER_CUBIC);
	}

	// Image pyramid
	void OneFrame::buildImgPyramid(const cv::Mat & src, std::vector<cv::Mat> & dst, const int & pyramidSize, const int & cvInterpolationFlag)
	{
		dst.resize(pyramidSize);
		dst[0] = src;
		for (size_t i = 1; i < pyramidSize; ++i)
			cv::resize(dst[i - 1], dst[i], cv::Size(0, 0), 0.5, 0.5, cvInterpolationFlag);
	}

	// GGM pyramids.
	void OneFrame::buildGGMPyramid()
	{
		cv::Mat tmpgrad; // added on 10/11/2022.
		// Do magnitude computation for each level.
		for (size_t i = 0; i < mvecMat_PyramidGray.size(); i++)
		{
			// Blur the grayscale image for better precision.
			cv::Mat oMImgBlur;
#ifdef WITH_BLUR // With Blur for calculating magitude.
			cv::GaussianBlur(mvecMat_PyramidGray[i], oMImgBlur, cv::Size(3, 3), 0);
#else  // Without blur
			oMImgBlur = mvecMat_PyramidGray[i].clone();  // Use the original UINT gray image instead of the 32FC1 gray image.
#endif
#ifdef WITH_SHARPEN // With sharpening.
			cv::addWeighted(mvecMat_PyramidGray[i], 1.5, oMImgBlur, -0.5, 0, oMImgBlur);
#endif

			if (Configuration::METRIC_LABEL==GGM)
				mGGM_Handler.gradientMagnitude(oMImgBlur, mMat_RGB, mvecMat_PyramidGrayGGM[i], mvecMat_PyramidGrayGGMPhase[i], tmpgrad, 0.5);
			else if (Configuration::METRIC_LABEL == LOG)
				mGGM_Handler.biqaLOG(oMImgBlur, mvecMat_PyramidGrayGGM[i]);
			else
			{
				std::cerr << "- Unknown feature metric!" << std::endl;
				exit(0);
			}
			//cv::imshow("mag img", m_pyramidMagnitude[i]);  // For displaying. uncommented.
			//cv::waitKey(0);
			//cv::Mat tmp = m_pyramidMagnitude[i].clone();
			//cv::circle(tmp, minL, 3, cv::Scalar(255, 0, 0), 2);
			//cv::circle(tmp, maxL, 3, cv::Scalar(255, 0, 0), 2);
			//cv::imshow("marked", tmp);
			//cv::waitKey(0);

			mvecMat_PyramidGrayGGM[i].convertTo(mvecMat_PyramidGrayGGM[i], CV_64F, 1.0 / 255, 0);  // Nomarlize into [1, 255].
			//m_pyramidMagnitude[i].convertTo(m_pyramidMagnitude[i], CV_32F, 1.0, 0);  // 
			mvecMat_PyramidGrayMaxGGMInBlk[i] = cv::Mat::zeros(mvecMat_PyramidGray[i].size(), mvecMat_PyramidGray[i].type());
			//cv::imshow("0 mat", m_pyramidMaxMagnitudeInBlk[i]);
			//cv::waitKey(0);


			std::vector<cv::Point2d> vP2dMaxBlkMag;  // To preserve points with local max magnitude of each block.
			corePixelInBlk(mvecMat_PyramidGrayGGM[i], mvecMat_PyramidDepth[i], mvecMat_PyramidGrayMaxGGMInBlk[i], cv::Rect2d(0, 0, mvecMat_PyramidGray32FC1[i].cols, mvecMat_PyramidGray32FC1[i].rows), vP2dMaxBlkMag, md_globalmin);

		} // end for
	} // end func


	// The maximum scalar of the metric values of a block.
	cv::Mat OneFrame::getMaxMagnitudeInBlk(int level) const // 31/08/2022.
	{
		return (mvecMat_PyramidGrayMaxGGMInBlk[level].clone()).reshape(1, mvecMat_PyramidGrayMaxGGMInBlk[level].rows * mvecMat_PyramidGrayMaxGGMInBlk[level].cols);
	}
	
	// Get image vector
	cv::Mat OneFrame::getImageVector(const int & level) const
	{
		return (mvecMat_PyramidGray32FC1[level].clone()).reshape(1, mvecMat_PyramidGray32FC1[level].rows * mvecMat_PyramidGray32FC1[level].cols);
	}

	// Pixel gradients in X per level
	cv::Mat OneFrame::getGradientX(const int & level) const
	{
		return (mvecMat_PyramidGrayIdx[level].clone()).reshape(1, mvecMat_PyramidGrayIdx[level].rows * mvecMat_PyramidGrayIdx[level].cols);
	}

	// Pixel gradients in Y per level
	cv::Mat OneFrame::getGradientY(const int & level) const
	{
		return (mvecMat_PyramidGrayIdy[level].clone()).reshape(1, mvecMat_PyramidGrayIdy[level].rows * mvecMat_PyramidGrayIdy[level].cols);
	}

	// Default block size is 8*8.
	// It generates a Mat object in UCHAR indicating local max magnitude of each block with white pixel (255).
	int OneFrame::corePixelInBlk(const cv::Mat & poMSrcMagImg, const cv::Mat & SrcDepth, cv::Mat & poMDstMaxMagInBlkImg, const cv::Rect2d& poOverlapped2dR, std::vector<cv::Point2d> & pvCVP, const float & globalmin, const int & blk_sz)
	{
		int N = 0;
		int x0 = poOverlapped2dR.x + 4, xN = x0 + poOverlapped2dR.width - 1;
		int y0 = poOverlapped2dR.y + 4, yN = y0 + poOverlapped2dR.height - 1;
		//double globalmin, globalmax;
		//cv::Point gminidx, gmaxidx;
		//cv::minMaxLoc(poMSrcMagImg, &globalmin, &globalmax, &gminidx, &gmaxidx);
		//while (globalmin < 1e-4)
		//	globalmin = globalmin * 10;
		for (int i = x0; i <= xN - blk_sz; i = i + blk_sz) // each column, x
		{
			for (int j = y0; j <= yN - blk_sz; j = j + blk_sz) // each row, y
			{
				cv::Mat tmpBlk = poMSrcMagImg(cv::Rect2d(i, j, blk_sz, blk_sz)).clone();
				cv::Mat tmpBlkDepth = SrcDepth(cv::Rect2d(i, j, blk_sz, blk_sz)).clone();
				double minVal, maxVal;
				cv::Point minIdx, maxIdx;
				cv::minMaxLoc(tmpBlk, &minVal, &maxVal, &minIdx, &maxIdx);
				cv::Scalar tmpScalar;

#ifdef PLAIN
				if (minVal >0 && (minVal != maxVal) && std::isfinite(tmpBlkDepth.at<float>(maxIdx))) // added on 08/11/2022.
#endif
				{
					pvCVP.push_back(maxIdx + cv::Point(i, j));
					poMDstMaxMagInBlkImg.at<uchar>(maxIdx + cv::Point(i, j)) = 255;
					N++;
				}
			}
		}
		return N;
	} // end func

	// Pixel-wise gradient in X
	void OneFrame::PixelWiseGradientX(cv::Mat& src, cv::Mat& dst)
	{
		dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f).clone();
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				if (x == 0)
					dst.at<float>(y, x) = (src.at<float>(y, x + 1) - src.at<float>(y, x));
				else if (x == src.cols - 1)
					dst.at<float>(y, x) = (src.at<float>(y, x) - src.at<float>(y, x - 1));
				else
					dst.at<float>(y, x) = (src.at<float>(y, x + 1) - src.at<float>(y, x - 1))*0.5;
			}
		}
	} // end func

	// Pixel-wise gradient in Y
	void OneFrame::PixelWiseGradientY(cv::Mat& src, cv::Mat& dst)
	{
		dst = cv::Mat(src.rows, src.cols, CV_32FC1, 0.f).clone();
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				if (y == 0)
					dst.at<float>(y, x) = (src.at<float>(y + 1, x) - src.at<float>(y, x));
				else if (y == src.rows - 1)
					dst.at<float>(y, x) = (src.at<float>(y, x) - src.at<float>(y - 1, x));
				else
					dst.at<float>(y, x) = (src.at<float>(y + 1, x) - src.at<float>(y - 1, x))*0.5;
			}
		}
	} // end func

	// Pixel-wise gradients computations
	void OneFrame::buildPixelWiseGradientPyramid()
	{
		int one(1);
		int zero(0);
		double scale = 0.5;

		PixelWiseGradientX(mvecMat_PyramidGray32FC1[0], mvecMat_PyramidGrayIdx[0]);
		PixelWiseGradientY(mvecMat_PyramidGray32FC1[0], mvecMat_PyramidGrayIdy[0]);

		buildImgPyramid(mvecMat_PyramidGrayIdx[0], mvecMat_PyramidGrayIdx, Configuration::PyramidLevel, cv::INTER_CUBIC);
		buildImgPyramid(mvecMat_PyramidGrayIdy[0], mvecMat_PyramidGrayIdy, Configuration::PyramidLevel, cv::INTER_CUBIC);
	}

	// Return depth per level
	cv::Mat OneFrame::getDepth(int level) const
	{
		return (mvecMat_PyramidDepth[level].clone()).reshape(1, mvecMat_PyramidDepth[level].rows * mvecMat_PyramidDepth[level].cols);
	}

	// Width of image each level
	int OneFrame::getWidthPerLevel(const int & level)
	{
		return mvecMat_PyramidGray32FC1.at(level).cols;   // We use the member of the 32FC1 
	}

	// Height of image each level
	int OneFrame::getHeightPerLevel(const int & level)
	{
		return mvecMat_PyramidGray32FC1.at(level).rows;   // We use here the same member to the width
	}

} // end space


