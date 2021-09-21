#include "preProcess.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace std::filesystem;

const string meta = ".\\cache.dat";

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

	imList->image = (ImageData*)malloc(sizeof(ImageData) * count);
	imList->n = 0;

	if (!imList->image) {
		return NULL;
	}

	for (const auto& file : directory_iterator(path)) {
		string at = file.path().string();
		if (at.find(meta) == string::npos) {
			strcpy(imList->image[imList->n].name, at.c_str());
			imList->image[imList->n++].hex = 0;
		}
	}

	return imList;

}

void calculateValues(ImageData* imData, cv::Mat image) {
	cv::Scalar mean = cv::mean(image);
	imData->B = mean[0];
	imData->G = mean[1];
	imData->R = mean[2];
	imData->gray = 0.299 * imData->R + 0.587 * imData->G + 0.114 * imData->B;
	imData->hex = imData->R * 256 * 256 + imData->G * 256 + imData->B;
}

ImageList* processImage(string path) {
	ImageList* imlist = readImages(path);

	for (int i = 0; i < imlist->n; i++) {
		cv::Mat image = cv::imread(imlist->image[i].name);
		calculateValues(&(imlist->image[i]), image);
	}

	insertionSort(imlist);

	return imlist;
}

bool saveCache(ImageList* imlist) {
	FILE* f = fopen(meta.c_str(), "wb+");

	if (f == NULL) {
		return false;
	}

	fprintf(f, "%d", imlist->n);
	fwrite(imlist->image, sizeof(ImageData), imlist->n, f);

	fclose(f);
	
	return true;
}

ImageList* readCache() {
	FILE* f = fopen(meta.c_str(), "rb");
	int n;

	if (f == NULL) {
		return NULL;
	}

	fscanf(f, "%d", &n);

	ImageList* imlist = (ImageList*)malloc(sizeof(ImageList));
	imlist->image = (ImageData*)malloc(sizeof(ImageData) * n);
	imlist->n = n;

	fread(imlist->image, sizeof(ImageData), n, f);

	fclose(f);

	return imlist;
}

void insertionSort(ImageList* imlist) {
	for (int i = 1; i < imlist->n; i++) {
		int vGray = imlist->image[i].gray;
		for (int j = i; j > 0 && vGray < imlist->image[j - 1].gray; j--) {
			ImageData aux = imlist->image[j];
			imlist->image[j] = imlist->image[j - 1];
			imlist->image[j - 1] = aux;
		}
	}
}