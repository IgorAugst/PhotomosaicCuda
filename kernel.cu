#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "preProcess.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define MAX 32

typedef struct RGB {
	int r;
	int g;
	int b;
};

typedef struct imageBlock {
	char name[MAXNAME];
};

__global__ void grayAvrgTestLine(unsigned char* input, int colorWidthStep, int n, RGB *resp) {
	int line = blockIdx.x;

	for (int i = 0; i < n; i++) {
		int index = line * colorWidthStep + (3 * i);
		
		resp[line].b += input[index];
		resp[line].g += input[index + 1];
		resp[line].r += input[index + 2];

	}

	resp[line].r /= n;
	resp[line].g /= n;
	resp[line].b /= n;

}

__global__ void grayAvrgTestRow(RGB* resp, int n) {
	for (int i = 1; i < n; i++) {
		resp[0].r += resp[i].r;
		resp[0].g += resp[i].g;
		resp[0].b += resp[i].b;
	}

	resp[0].r /= n;
	resp[0].g /= n;
	resp[0].b /= n;
}

__global__ void grayAvrgKernel(unsigned char* input, int colorWidthStep, int blockWidth, int blockHeight, int imgWidth, int imgHeight) {
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int sx = blockX * blockWidth;
	int sy = blockY * blockHeight;

	int fx = sx + blockWidth;
	int fy = sy + blockHeight;

	if (fx > imgWidth) {
		fx = imgWidth;
	}

	if (fy > imgHeight) {
		fy = imgHeight;
	}
	
	RGB color;
	color.r = 0;
	color.g = 0;
	color.b = 0;

	for (int y = 0; y < fy; y++) {
		for (int x = 0; x < fx; x++) {
			int index = y * colorWidthStep + (3 * x);

			color.b += input[index];
			color.g += input[index + 1];
			color.r += input[index + 2];
		}
	}

	int n = imgWidth * imgHeight;

	color.b /= n;
	color.g /= n;
	color.r /= n;

	printf("R: %d, G: %d, B: %d", color.r, color.g, color.b);

}


void cacheTest() {
	ImageList* imlist = processImage("D:\\igora\\Documents\\Code\\Photomosaic\\testes");

	bool status = saveCache(imlist);

	cout << (status ? "salvo" : "erro") << endl;

	free(imlist->image);
	free(imlist);

	imlist = readCache();

	cout << (imlist != NULL ? "lido" : "erro") << endl;

	getchar();
}

void averageTest() {
	Mat image = imread("D:\\igora\\Pictures\\teste.png");
	namedWindow("teste");
	imshow("teste", image);

	unsigned char *dImage;
	int size = image.step * image.rows;

	cudaMalloc<unsigned char>(&dImage, size);
	cudaMemcpy(dImage, image.ptr(), size, cudaMemcpyHostToDevice);

	dim3 blocks(image.rows);
	
	RGB* values;

	cudaMallocManaged(&values, sizeof(RGB) * image.rows);

	grayAvrgTestLine<<<blocks, 1>>>(dImage, image.step, image.cols, values);

	cudaDeviceSynchronize();

	grayAvrgTestRow << <1, 1 >> > (values, image.rows);

	cudaDeviceSynchronize();

	printf("R: %d, G: %d, B: %d", values[0].r, values[0].g, values[0].b);

	printf("\nteste loop\n");

	grayAvrgKernel<<<1,1>>>(dImage, image.step, image.cols, image.rows, image.cols, image.rows);

	cudaDeviceSynchronize();

	waitKey(0);

}

int main(int argc, char** argv) {

	//cacheTest();
	averageTest();

}