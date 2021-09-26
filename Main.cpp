#include "preProcess.h"
#include "kernel.h"
#include <argparse/argparse.hpp>
#include <string>

/*
Argumentos da linha de comando
Entrada
Saida
quantidade de blocos
resolução das imagens
atualizar o cache 
preto e branco (bool)
*/

int main(int argc, char* argv[]) {
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

	printf("Calculando correspondencias...\n");
	ImageList *structure = Average(inputImage, imList, x);

	int res = program.get<int>("--res");
	dim3 resolution(res, res);

	int y = structure->n / x;

	dim3 finalSize(x * res, y * res);

	Mat finalImage(finalSize.x, finalSize.y, CV_8UC3);

	printf("Gerando Mosaico...\n");

	GenerateImage(structure, imList, x, resolution, finalSize, &finalImage);

	namedWindow("final");
	imshow("final", finalImage);

	printf("Fim.\n");

	waitKey();

}