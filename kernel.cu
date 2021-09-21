#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "preProcess.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_set>

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

	printf("R: %d, G: %d, B: %d\n", color.r, color.g, color.b);

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

	printf("%s\n", imData[lowIndex].name);
}

__global__ void FillImageKernel(unsigned char* input, int colorWidthStep, dim3 inputSize, ImageData* imData, dim3 quant, dim3 subSize, unsigned char* subImg, int subStep, int hex) {
	int pixX = threadIdx.x + blockIdx.x * blockDim.x;
	int pixY = threadIdx.y + blockIdx.y * blockDim.y;

	int bImgX = pixX / (inputSize.x / quant.x);
	int bImgy = pixY / (inputSize.y / quant.y);

	int blockIndex = bImgy * quant.x + bImgX;

	if (imData[blockIndex].hex != hex) {
		return;
	}

	//REFAZER TUDO ISSO AQUI QUE EU NÃO TO ENTENDENDO MAIS NADA

	float rX = (float)subSize.x / (float)inputSize.x;
	float rY = (float)subSize.y / (float)inputSize.y;

	int pixSubX = pixX / rX;
	int pixSubY = pixY / rY;

	int outIndex = pixY * colorWidthStep + (3 * pixX);
	int subIndex = pixSubY * subStep + (3 * pixSubX);

	
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

void bestImageTest() {
	Mat image = imread("D:\\igora\\Pictures\\teste.png");
	unsigned char* dImage;

	int size = image.rows * image.step;

	cudaMalloc<unsigned char>(&dImage, size);
	cudaMemcpy(dImage, image.ptr(), size, cudaMemcpyHostToDevice);

	dim3 block(2, 2);
	
	int blockWidth = image.cols / block.x;
	int blockHeight = image.rows / block.y;

	ImageList* imgList = processImage("D:\\igora\\Documents\\Code\\Photomosaic\\testes");
	ImageData* imgData;

	cudaMalloc<ImageData>(&imgData, sizeof(ImageData) * imgList->n);
	cudaMemcpy(imgData, imgList->image, sizeof(ImageData) * imgList->n, cudaMemcpyHostToDevice);

	ImageData* out;

	cudaMallocManaged<ImageData>(&out, sizeof(ImageData) * block.x * block.y);

	AvrgKernel<<<block, 1>>>(dImage, image.step, image.cols, image.rows, imgData, imgList->n, out);

	cudaDeviceSynchronize();

	waitKey();

	//------------------------------------------------------
	//Leitura das imagens e distribuição

	int quantBlock = block.x * block.y;
	unsigned char** imgArray;
	imgArray = (unsigned char**)malloc(sizeof(char*) * quantBlock);
	
	unordered_set<String> usedImg;

	for (int i = 0; i < quantBlock; i++) {
		if (usedImg.find(out[i].name) == usedImg.end()) {
			Mat imgAux = imread(out[i].name);
			int size = imgAux.rows * imgAux.step;

			cudaMalloc<unsigned char>(&imgArray[i], size);
			cudaMemcpy(imgArray[i], imgAux.ptr(), size, cudaMemcpyHostToDevice);

			//kernel

			usedImg.insert(out[i].name);
		}



	
		//TODO: utilizar hashmap para não ler a mesma imagem mais de uma vez
		//TODO: kernel para preencher a imagem
		//TODO: definir o tamanho da imagem
		//TODO: alocar imagem final na gpu
		//TODO: realizar calculo para encontrar os pixels correspondentes quando cortar a imagem (cortar pelo canto ou centro)
			
	}

}

int main(int argc, char** argv) {

	//cacheTest();
	//averageTest();
	bestImageTest();

}