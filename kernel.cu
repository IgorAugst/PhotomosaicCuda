#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "preProcess.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <time.h>

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

__global__ void AvrgKernel(unsigned char* input, int colorWidthStep, int imgWidth, int imgHeight, ImageData* imData, int imgQuant, ImageData* out) {
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int blockWidth = imgWidth / gridDim.x;
	int blockHeight = imgHeight / gridDim.y;

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

	for (int y = sy; y < fy; y++) {
		for (int x = sx; x < fx; x++) {
			int index = y * colorWidthStep + (3 * x);

			color.b += input[index];
			color.g += input[index + 1];
			color.r += input[index + 2];
		}
	}

	int n = blockWidth * blockHeight;

	color.b /= n;
	color.g /= n;
	color.r /= n;

	RGB diff;
	diff.r = color.r - imData[0].R;
	diff.g = color.g - imData[0].G;
	diff.b = color.b - imData[0].B;

	int lowDist = diff.r * diff.r + diff.g * diff.g + diff.b * diff.b;
	int lowIndex = 0;

	for (int i = 1; i < imgQuant; i++) {
		diff.r = color.r - imData[i].R;
		diff.g = color.g - imData[i].G;
		diff.b = color.b - imData[i].B;

		int distAux = diff.r * diff.r + diff.g * diff.g + diff.b * diff.b;

		if (distAux < lowDist) {
			lowDist = distAux;
			lowIndex = i;
		}
	}

	int gridWidth = gridDim.x;

	int index = blockY * gridWidth + blockX;

	
	out[index] = imData[lowIndex];

	//printf("%s\n", out[index].name);
}

__global__ void FillImageKernel(unsigned char* output, int outputStep, dim3 outputSize, ImageData* imData, dim3 quantBlock, dim3 blockImgSize, unsigned char* blockImg, int blockStep, int hex) {
	
	int outX = blockDim.x * blockIdx.x + threadIdx.x;
	int outY = blockDim.y * blockIdx.y + threadIdx.y;

	dim3 outputPartialSize(outputSize.x * quantBlock.x, outputSize.y * quantBlock.y);

	if (outX > outputPartialSize.x || outY > outputPartialSize.y) {
		return;
	}

	int outputTotalSize = outputPartialSize.x * outputPartialSize.y;

	int outputIndex = outY * outputStep + (3 * outX);

	//printf("pixel %d\n", outputIndex);

	int blockImgY = (float)outY / (float)outputSize.y;
	int blockImgX = (float)outX / (float)outputSize.x;

	int blockImgIndex = blockImgY * quantBlock.x + blockImgX;

	
	//printf("%d %d\n", outX, outY);

	if (imData[blockImgIndex].hex != hex) {
		return;
	}

	//printf("i: %d\n", blockImgIndex);

	float rX = (float)outputSize.x / ((float)blockImgSize.x);
	float rY = (float)outputSize.y / ((float)blockImgSize.y);

	int pixSubX = (outX % outputSize.x) / rX;
	int pixSubY = (outY % outputSize.y) / rY;

	
	int subIndex = pixSubY * blockStep + (3 * pixSubX);

	output[outputIndex] = blockImg[subIndex];
	output[outputIndex + 1] = blockImg[subIndex + 1];
	output[outputIndex + 2] = blockImg[subIndex + 2];

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

/*
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

	AvrgKernel<<<2,1>>>(dImage, image.step, image.cols, image.rows);

	cudaDeviceSynchronize();

	waitKey(0);

}
*/

ImageData* Average(Mat img, ImageList* imList, int x) {
	float ratio = (float)img.cols / (float)img.rows;
	int yBlocks = img.rows / ratio;
	dim3 blockKernel(x, y);

	unsigned char* dImage;
	int size = img.rows * img.step;

	cudaMalloc<unsigned char>(&dImage, size);
	cudaMemcpy(dImage, img.ptr(), size, cudaMemcpyHostToDevice); //aloca e copia a imagem para gpu

	ImageData* imData;
	cudaMalloc<ImageData>(&imData, sizeof(ImageData) * imList->n);
	cudaMemcpy(imData, imList->image, sizeof(ImageData) * imList->n, cudaMemcpyHostToDevice); //aloca e copia o cache das imagens

	ImageData* outData;
	cudaMalloc<ImageData>(&outData, sizeof(ImageData) * x * y); //aloca os dados de saida

	AvrgKernel<<<blockKernel, 1>>>(dImage, img.step, img.cols, img.rows, imData, imList->n, outData);

	cudaDeviceSynchronize();

	ImageData* hostData;
	hostData = (ImageData*)malloc(sizeof(ImageData) * x * y);

	cudaMemcpy(hostData, outData, sizeof(ImageData) * x * y, cudaMemcpyDeviceToHost); // copia os dados para o host

	cudaFree(outData);
	cudaFree(imData);
	cudaFree(dImage);

	return hostData;

}

void bestImageTest() {
	clock_t begin = clock();

	Mat image = imread("D:\\igora\\Downloads\\Screenshot_20210318-082317.jpg");
	unsigned char* dImage;

	int size = image.rows * image.step;

	cudaMalloc<unsigned char>(&dImage, size);
	cudaMemcpy(dImage, image.ptr(), size, cudaMemcpyHostToDevice);

	dim3 block(200, 200);
	
	int blockWidth = image.cols / block.x;
	int blockHeight = image.rows / block.y;

	//ImageList* imgList = processImage("D:\\igora\\Documents\\Code\\Photomosaic\\images");
	//saveCache(imgList);
	ImageList* imgList = readCache();
	ImageData* imgData;

	cudaMalloc<ImageData>(&imgData, sizeof(ImageData) * imgList->n);
	cudaMemcpy(imgData, imgList->image, sizeof(ImageData) * imgList->n, cudaMemcpyHostToDevice);

	ImageData* out;
	ImageData* outDevData;

	cudaMalloc<ImageData>(&outDevData, sizeof(ImageData) * block.x * block.y);

	out = (ImageData*)malloc(sizeof(ImageData) * block.x * block.y);

	printf("calculando valores...\n");

	AvrgKernel<<<block, 1>>>(dImage, image.step, image.cols, image.rows, imgData, imgList->n, outDevData);

	cudaDeviceSynchronize();

	cudaMemcpy(out, outDevData, sizeof(ImageData) * block.x * block.y, cudaMemcpyDeviceToHost);

	//------------------------------------------------------
	//Leitura das imagens e distribuição

	int quantBlock = block.x * block.y;
	unsigned char** imgArray;
	imgArray = (unsigned char**)malloc(sizeof(char*) * quantBlock);
	
	set<int> usedImg;

	dim3 outSize(50, 50);

	dim3 outImageSize(outSize.x * block.x, outSize.y * block.y);

	Mat outImg(outImageSize.y, outImageSize.x, CV_8UC3);
	unsigned char* outDev;
	int outImgSize = outImg.step * outImg.rows;

	cudaMalloc<unsigned char>(&outDev, outImgSize);

	printf("Preenchendo a imagem...\n");

	for (int i = 0; i < quantBlock; i++) {
		if (usedImg.find(out[i].hex) == usedImg.end()) {
			Mat imgAux = imread(out[i].name);
			int size = imgAux.rows * imgAux.step;

			cudaMalloc<unsigned char>(&imgArray[i], size);
			cudaMemcpy(imgArray[i], imgAux.ptr(), size, cudaMemcpyHostToDevice);

			dim3 blockImgSize(imgAux.cols, imgAux.rows);

			dim3 threads(MAX, MAX);

			dim3 blockKernel(outImageSize.x / MAX, outImageSize.y / MAX);

			blockKernel.x++;
			blockKernel.y++;

			FillImageKernel << <blockKernel, threads>> > (outDev, outImg.step, outSize, outDevData, block, blockImgSize, imgArray[i], imgAux.step, out[i].hex);

			usedImg.insert(out[i].hex);
		}

			
	}


	cudaDeviceSynchronize();

	printf("salvando imagem...\n");

	cudaMemcpy(outImg.ptr(), outDev, outImgSize, cudaMemcpyDeviceToHost);

	/*
	namedWindow("vai");
	imshow("vai", outImg);
	*/

	imwrite("D:\\igora\\Pictures\\PhotoCuda\\teste.jpg", outImg);

	clock_t end = clock();

	double timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("tempo gasto: %.2fs\n", timeSpent);

	waitKey();
}