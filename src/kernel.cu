#include "kernel.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nArquivo: %s\n\nLinha: %d\n\nMotivo: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
		system("pause");
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

#define MAX 32

#define CUDA_HOST_DEVICE "Cuda MemCpy host to device"
#define CUDA_MALLOC "Cuda Malloc"
#define CUDA_DEVICE_HOST "Cuda MemCpy device to host"
#define CUDA_KERNEL "Kernel launch"

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

	float blockWidth = (float)imgWidth / (float)gridDim.x;
	float blockHeight = (float)imgHeight / (float)gridDim.y;

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
	
	unsigned int outX = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int outY = blockDim.y * blockIdx.y + threadIdx.y;

	dim3 outputPartialSize(outputSize.x * quantBlock.x, outputSize.y * quantBlock.y);

	if (outX >= outputPartialSize.x || outY >= outputPartialSize.y) {
		return;
	}

	unsigned int outputTotalSize = outputPartialSize.x * outputPartialSize.y;

	unsigned int outputIndex = outY * outputStep + (3 * outX);

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

	unsigned int pixSubX = (outX % outputSize.x) / rX;
	unsigned int pixSubY = (outY % outputSize.y) / rY;

	
	unsigned int subIndex = pixSubY * blockStep + (3 * pixSubX);

	output[outputIndex] = blockImg[subIndex];
	output[outputIndex + 1] = blockImg[subIndex + 1];
	output[outputIndex + 2] = blockImg[subIndex + 2];

}

__global__ void ToGrayScaleKernel(unsigned char* input, int inputStep, dim3 imageDim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= imageDim.x || y >= imageDim.y) {
		return;
	}

	int index = y * inputStep + (3 * x);

	int gray = 0;

	gray = (input[index] * 0.11) + (input[index + 1] * 0.59 )+ (input[index + 2] * 0.3);

	input[index] = gray;
	input[index + 1] = gray;
	input[index + 2] = gray;
}

ImageList* Average(Mat img, ImageList* imList, int x) {
	float ratio = (float)img.cols / (float)img.rows;
	int yBlocks = ceil(x / ratio);
	dim3 blockKernel(x, yBlocks);

	unsigned char* dImage;
	int size = img.rows * img.step;

	SAFE_CALL(cudaMalloc<unsigned char>(&dImage, size), CUDA_MALLOC);
	SAFE_CALL(cudaMemcpy(dImage, img.ptr(), size, cudaMemcpyHostToDevice), CUDA_HOST_DEVICE); //aloca e copia a imagem para gpu

	ImageData* imData;
	SAFE_CALL(cudaMalloc<ImageData>(&imData, sizeof(ImageData) * imList->n), CUDA_MALLOC);
	SAFE_CALL(cudaMemcpy(imData, imList->image, sizeof(ImageData) * imList->n, cudaMemcpyHostToDevice), CUDA_HOST_DEVICE); //aloca e copia o cache das imagens

	ImageData* outData;
	SAFE_CALL(cudaMalloc<ImageData>(&outData, sizeof(ImageData) * x * yBlocks), CUDA_MALLOC); //aloca os dados de saida

	AvrgKernel<<<blockKernel, 1>>>(dImage, img.step, img.cols, img.rows, imData, imList->n, outData);

	SAFE_CALL(cudaDeviceSynchronize(), "Kernel launch");

	ImageData* hostData;
	hostData = (ImageData*)malloc(sizeof(ImageData) * x * yBlocks);

	SAFE_CALL(cudaMemcpy(hostData, outData, sizeof(ImageData) * x * yBlocks, cudaMemcpyDeviceToHost), CUDA_DEVICE_HOST); // copia os dados para o host

	cudaFree(outData);
	cudaFree(imData);
	cudaFree(dImage);

	ImageList* outList = (ImageList*)malloc(sizeof(ImageList));
	outList->image = hostData;
	outList->n = x * yBlocks;

	return outList;

}

void GenerateImage(ImageList* structure, ImageList* cache, int x, dim3 resDim,dim3 finalImageSize, Mat *finalImage, bool grayscale, void(*progressCallback)(int, int)) {
	int y = structure->n / x;

	dim3 blockQuant(x, y);
	//dim3 finalImageSize(x * res, y * res);

	//dim3 resDim(res, res);

	ImageData *devData;

	SAFE_CALL(cudaMalloc<ImageData>(&devData, sizeof(ImageData) * structure->n), CUDA_MALLOC);
	SAFE_CALL(cudaMemcpy(devData, structure->image, sizeof(ImageData) * structure->n, cudaMemcpyHostToDevice), CUDA_HOST_DEVICE);

	//Mat finalImage(x, y, CV_8UC3);
	unsigned char* dFinalImage;

	unsigned int sizeFinal = finalImage->rows * finalImage->step;

	cudaMalloc<unsigned char>(&dFinalImage, sizeFinal);

	unsigned char* dImageAux;

	dim3 blockKernel(finalImageSize.x / MAX, finalImageSize.y / MAX);
	blockKernel.x++;
	blockKernel.y++;
	dim3 threads(MAX, MAX);	

	set<int> usedImage;

	for (int i = 0; i < structure->n; i++) {
		if (usedImage.find(structure->image[i].hex) != usedImage.end()) {
			continue;
		}
		Mat imgAux = imread(structure->image[i].name);
		int size = imgAux.step * imgAux.rows;

		cudaMalloc<unsigned char>(&dImageAux, size);
		cudaMemcpy(dImageAux, imgAux.ptr(), size, cudaMemcpyHostToDevice);

		dim3 dimAux(imgAux.cols, imgAux.rows);

		FillImageKernel<<<blockKernel, threads>>>(dFinalImage, finalImage->step, resDim, devData, blockQuant, dimAux, dImageAux, imgAux.step, structure->image[i].hex);

		cudaDeviceSynchronize();

		cudaFree(dImageAux);

		usedImage.insert(structure->image[i].hex);

		if (i % 10 == 0) {
			(*progressCallback)(i, structure->n);
		}

	}

	(*progressCallback)(100, 100);

	if (grayscale) {
		ToGrayScaleKernel<<<blockKernel, threads>>>(dFinalImage, finalImage->step, finalImageSize);
	}

	SAFE_CALL(cudaMemcpy(finalImage->ptr(), dFinalImage,sizeFinal, cudaMemcpyDeviceToHost), "Image generation");

	return;
}