#include "preProcess.h"
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std::filesystem;

const string meta = "cache.dat";

ImageList* readImages(string path) {
	int count = 0;

	for (const auto & file : directory_iterator(path)) {
		string at = file.path().string();
		if (at.find(meta) == string::npos) {
			count++;
		}
	}

	ImageList *imList = (ImageList*)malloc(sizeof(ImageList));

	if (!imList) {
		return NULL;
	}

	imList->image = new ImageData[count];
	imList->n = 0;

	if (!imList->image) {
		return NULL;
	}

	for (const auto& file : directory_iterator(path)) {
		string at = file.path().string();
		if (at.find(meta) == string::npos) {
			imList->image[imList->n++].name = at;
		}
	}

	return imList;

}

void calculateValues(ImageData* imData, cv::Mat image) {
	cv::Scalar mean = cv::mean(image);
}

void processImage(ImageList* imlist) {
	for (int i = 0; i < imlist->n; i++) {
		cv::Mat image = cv::imread(imlist->image[i].name);
	}
}