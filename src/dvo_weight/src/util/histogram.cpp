/**
 *  This file is part of dvo.
 *
 *  Copyright 2012 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
 *  For more information see <http://vision.in.tum.de/data/software/dvo>.
 *
 *  dvo is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  dvo is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <dvo/util/histogram.h>




int getNumberOfBins(float min, float max, float binWidth)
{
  return (int)((max - min + 1) / binWidth);
}

void compute1DHistogram(const cv::Mat& data, cv::Mat& histogram, float min, float max, float binWidth)
{
  cv::Mat mask;

  cv::Mat images[] = { data };
  int channels[] = { 0 };
  int nbins[] = { getNumberOfBins(min, max, binWidth) };

  float range[] = { min, max };
  const float* ranges[] = { range };

  // seems to ignore nan values
  cv::calcHist(images, 1, channels, mask, histogram, 1 /*dimensions*/, nbins/*number of bins*/, ranges /*range*/, true /*uniform*/, false /*accumulate*/);
}

float computeMedianFromHistogram(const cv::Mat& histogram, float min, float max)
{
  float total_half = countElementsInHistogram(histogram) / 2.0f;
  const float* histogram_ptr = histogram.ptr<float>();

  float median = 0.0f;
  float acc = 0.0f;

  for(size_t idx = 0; idx < histogram.size().area(); ++idx, ++histogram_ptr)
  {
    acc += *histogram_ptr;

    if(acc > total_half)
    {
      median = idx;
      break;
    }
  }

  return median + min;
}

int countElementsInHistogram(const cv::Mat& histogram)
{
  int num = 0;
  const float* histogram_ptr = histogram.ptr<float>();

  for(size_t idx = 0; idx < histogram.size().area(); ++idx, ++histogram_ptr)
  {
    num += (int) *histogram_ptr;
  }

  return num;
}


