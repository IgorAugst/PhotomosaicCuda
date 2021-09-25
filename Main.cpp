#include "preProcess.h"
#include <argparse/argparse.hpp>


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


}