#ifndef JANICE_IO_OPENCV_FM_H
#define JANICE_IO_OPENCV_FM_H

#include <janice_io_opencv.h>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

JaniceError janice_io_opencv_create_frommat(std::vector<cv::Mat> ims, JaniceMediaIterator *_it);

#endif
