#pragma once
#include <string>

using namespace std;

typedef struct ImageData {
	string name;
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

ImageList* readImages(int quant, string path);