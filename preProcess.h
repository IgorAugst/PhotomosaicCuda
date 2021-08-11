#pragma once
#include <string>

using namespace std;

#define MAXNAME 100

typedef struct ImageData {
	char name[MAXNAME];
	int gray;
	int R;
	int G;
	int B;
	int quant;
};

typedef struct ImageList {
	ImageData* image;
	int n;
};

ImageList* processImage(string path);

bool saveCache(ImageList* imlist);

void insertionSort(ImageList* imlist);

ImageList* readCache();