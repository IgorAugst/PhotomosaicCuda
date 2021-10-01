#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "preProcess.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <time.h>

using namespace std;
using namespace cv;

ImageList* Average(Mat img, ImageList* imList, int x);

void GenerateImage(ImageList* structure, ImageList* cache, int x, dim3 resDim, dim3 finalImageSize, Mat* finalImage, bool grayscale, void(*progressCallback)(int, int));