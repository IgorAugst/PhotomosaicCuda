#include <Windows.h>
#include <shellapi.h>

#include "preProcess.h"
#include "kernel.h"
#include <argparse/argparse.hpp>
#include <string>
#include <chrono>
#include <locale.h>

using namespace std::chrono;

/*
Argumentos da linha de comando
Entrada
Saida
quantidade de blocos
resolução das imagens
atualizar o cache 
preto e branco (bool)
*/

void progressBar(int n, int max);

int main(int argc, char* argv[]) {
	setlocale(LC_ALL, "");

	auto start = high_resolution_clock::now();

	SetConsoleOutputCP(CP_UTF8);

	argparse::ArgumentParser program("Photomosaic");
	program.add_argument("-i", "--input").help("Imagem de entrada").required();
	program.add_argument("-o", "--output").help("Imagem de saída").required();
	program.add_argument("-q", "--quant").help("Quantidade de blocos para compor a imagem final (eixo X)").default_value(100).scan<'i',int>();
	program.add_argument("-r", "--res").help("Resolução de cada um dos blocos").default_value(30).scan<'i',int>();
	program.add_argument("-u", "--update").help("define o diretorio de imagens e atualiza o cache").default_value(".");
	program.add_argument("-g", "--gray").help("Gera a imagem em preto e branco").default_value(false).implicit_value(true);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	bool update = program.is_used("--update");

	string imagesDir;

	if (update) {
		imagesDir = program.get<string>("--update");
	}

	ImageList* imList;

	printf("Gerando/lendo cache...\n");

	if (update) {
		imList = processImage(imagesDir);
		saveCache(imList);
	}
	else {
		imList = readCache();
	}

	Mat inputImage = imread(program.get<string>("--input"));

	int x = program.get<int>("--quant");

	printf("Calculando correspondências...\n");
	ImageList *structure = Average(inputImage, imList, x);

	int res = program.get<int>("--res");
	dim3 resolution(res, res);

	int y = structure->n / x;

	dim3 finalSize(x * res, y * res);

	Mat finalImage(finalSize.y, finalSize.x, CV_8UC3);

	printf("Gerando Mosaico...\n");

	bool grayscale = program.get<bool>("--gray");

	GenerateImage(structure, imList, x, resolution, finalSize, &finalImage, grayscale, progressBar);

	string outDir = program.get<string>("--output");

	imwrite(outDir, finalImage);

	ShellExecute(NULL, "open", outDir.c_str(), NULL, NULL, SW_SHOW);

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(stop - start);

	float secDuration = (float)duration.count() / 1000.0;

	printf("Execução finalizada em: %.2f segundos",secDuration);

	waitKey();

}

void progressBar(int n, int max) {
	float x = (float)n / (float)max;
	x *= 100;

	printf("[");

	for (int i = 0; i <= 100; i++) {
		if (i <= x) {
			printf("#");
		}
		else {
			printf(" ");
		}
	}

	printf("]\r");

	if (n == max) {
		printf("\n");
	}
}