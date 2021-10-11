#pragma once
#include <Windows.h>
#include <shellapi.h>

#include <string>

using namespace std;

#define MAXNAME 100

typedef struct ImageData {
	char name[MAXNAME];
	int gray;
	int R;
	int G;	
	int B;
	int hex;
};

typedef struct ImageList {
	ImageData* image;
	int n;
};

ImageList* processImage(string path);

bool saveCache(ImageList* imlist);

void insertionSort(ImageList* imlist);

ImageList* readCache();

void start();