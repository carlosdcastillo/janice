#ifndef JANICE_IO_OPENCV_FM_H
#define JANICE_IO_OPENCV_FM_H

#include <janice_io_opencv.h>

#include <opencv2/highgui/highgui.hpp>

JaniceError janice_io_opencv_create_frommat(cv::Mat im, JaniceMediaIterator *_it);

#endif