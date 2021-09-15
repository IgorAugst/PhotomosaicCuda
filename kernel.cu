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

__global__ void grayAvrgTestLine(unsigned char* input, int colorWidthStep, int n) {
	int id = threadIdx.x;
	int step = 1;
	int numThread = blockDim.x;
	int line = blockIdx.x;

	while (numThread > 0) {
		if (id < numThread) {
			int first = id * step * 2;
			int sec = first + step;

			if (first < n && sec < n) {
				const int pix1 = line * colorWidthStep + (3 * first);
				const int pix2 = line * colorWidthStep + (3 * sec);
				
				int blue1 = input[pix1];
				int green1 = input[pix1 + 1];
				int red1 = input[pix1 + 2];

				int blue2 = input[pix2];
				int green2 = input[pix2 + 1];
				int red2 = input[pix2 + 2];

				input[pix1] = (blue1 + blue2);
				input[pix1 + 1] = (green1 + green2);
				input[pix1 + 2] = (red1 + red2);
				
			}
		}

		step *= 2;
		numThread % 2 == 0 || numThread == 1 ? numThread /= 2 : numThread = (numThread + 1) / 2;
	}

	__syncthreads();
}

__global__ void grayAvrgTestRow(unsigned char* input, int colorWidthStep, int n, int* resp) {
	int id = threadIdx.x;
	int step = 1;
	int numThread = blockDim.x;
	int row = blockIdx.x;

	while (numThread > 0) {
		if (id < numThread) {
			int first = id * step * 2;
			int sec = first + step;

			if (first < n && sec < n) {
				const int pix1 = first * colorWidthStep + (3 * row);
				const int pix2 = sec * colorWidthStep + (3 * row);

				int blue1 = input[pix1];
				int green1 = input[pix1 + 1];
				int red1 = input[pix1 + 2];

				int blue2 = input[pix2];
				int green2 = input[pix2 + 1];
				int red2 = input[pix2 + 2];

				input[pix1] = (blue1 + blue2);
				input[pix1 + 1] = (green1 + green2);
				input[pix1 + 2] = (red1 + red2);

			}
		}

		step *= 2;
		numThread % 2 == 0 || numThread == 1 ? numThread /= 2 : numThread = (numThread + 1) / 2;
	}

	(*resp) = input[0];
	__syncthreads();
}

__global__ void grayAvrgTestLineLoop(unsigned char* input, int colorWidthStep, int n, RGB *resp) {
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

__global__ void grayAvrgKernel(unsigned char* input, int width, int height, int colorWidthStep) {
	
}

void grayAvrgHelper(Mat* image, int bx, int by) {
	dim3 threads(MAX, MAX);
	dim3 blocks(0, 0);
	blocks.x = ceil((float)image->rows / (float)MAX);
	blocks.y = ceil((float)image->cols / (float)MAX);

	if (bx > blocks.x) {
		blocks.x = bx;
		threads.x = ceil((float)image->rows / (float)blocks.x);
	}

	if (by > blocks.y) {
		blocks.y = by;
		threads.y = ceil((float)image->cols / (float)blocks.y);
	}
	
	unsigned char* imDevice;
	int size = image->rows * image->cols;

	cudaMalloc(&imDevice, size);
	cudaMemcpy(imDevice, image->ptr(), size, cudaMemcpyHostToDevice);

	grayAvrgKernel << <blocks, threads >> > (imDevice, image->rows, image->cols, image->step);

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

	grayAvrgTestLineLoop<<<blocks, 1>>>(dImage, image.step, image.cols, values);

	cudaDeviceSynchronize();

	printf("B: %d, G: %d, R: %d", values[0].r, values[0].g, values[0].b);

}

int main(int argc, char** argv) {

	//cacheTest();
	averageTest();

}