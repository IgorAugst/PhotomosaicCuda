#include "preProcess.h"
#include <stdio.h>
#include <iostream>
#include <filesystem>

using namespace std::filesystem;

const string meta = "cache.dat";

ImageList* readImages(int quant, string path) {
	int count = 0;

	for (const auto & file : directory_iterator(path)) {
		string at = file.path().string();
		if (at != meta) {
			count++;
		}
	}

	ImageList *imList = (ImageList*)malloc(sizeof(ImageList));

	if (!imList) {
		return NULL;
	}

	imList->image = (ImageData*)malloc(count * sizeof(ImageData));
	imList->n = 0;

	if (!imList->image) {
		return NULL;
	}

	for (const auto& file : directory_iterator(path)) {
		string at = file.path().string();
		imList->image[imList->n++].name = at;
	}

}

void processImage() {

}