/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <fstream>
#include <iostream>
using namespace std;

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code

	/*Terei que fazer uma coisa semelhante ao HW3, em que uso uma matriz de threads * bins para
	cada thread ir colocando um elemento numa posição da matriz de cada vez que o elemento corresponde a esse bin.
	(por exemplo, thread 1 encontra um elemento que pertence ao bin 3 --> matriz[1][3] ++ ou matriz[1][3] = 1.
	Depois, é preciso, para cada linha (ou coluna, conforme a matriz esteja feita) fazer um reduce, para calcular o numero
	total de elementos num bin. 

	O algoritmo em si deve ser bastante facil, o dificil vai ser corre-lo o mais depressa possivel!
	*/

}

__global__
void reduceBin(unsigned int* d_bucketValues, int elementsToProcess, unsigned int* const d_out, int i, int numThreads){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int d = i * numThreads;
	for (int s = elementsToProcess / 2; s > 0; s = s / 2){
		if (threadIdx.x < s){
			d_bucketValues[tid + d] = d_bucketValues[tid + d] + d_bucketValues[tid + s + d];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0){
		d_out[i] = d_bucketValues[d];
	}
}

__global__
void calculateBins(unsigned int* d_binMatrix, const unsigned int* const d_vals, const unsigned int numBins, const unsigned int numElems, int numThreads){
	//cada thread processa 1 000 elementos. 
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int d_vals_pos = tid * 10000;
	if (d_vals_pos >= numElems){
		return;
	}
	int k, j;
	for (int i = 0; i < 10000; ++i){
		k = d_vals[d_vals_pos + i] * numThreads + tid;
		d_binMatrix[d_vals[d_vals_pos + i] * numThreads + tid] = d_binMatrix[d_vals[d_vals_pos + i] * numThreads + tid] + 1;
		j = d_binMatrix[d_vals[d_vals_pos + i] * numThreads + tid];
	}

	return;
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

	//cada thread processará 1 000 elementos. 
	int threadsPerBlock = 1024;
	int numBlocks = 1;

	unsigned int* d_binMatrix;
	checkCudaErrors(cudaMalloc(&d_binMatrix, sizeof(unsigned int) * numBins * threadsPerBlock * numBlocks));
	checkCudaErrors(cudaMemset(d_binMatrix, 0, sizeof(unsigned int) * numBins * threadsPerBlock * numBlocks));

	unsigned int* d_reduceBin_out;
	checkCudaErrors(cudaMalloc(&d_reduceBin_out, sizeof(int)));
	unsigned int* d_bucketValues;
	checkCudaErrors(cudaMalloc(&d_bucketValues, sizeof(unsigned int) * threadsPerBlock * numBlocks));
	calculateBins << <numBlocks, threadsPerBlock >> >(d_binMatrix, d_vals, numBins, numElems, threadsPerBlock);

	int val;
	ofstream file;
	file.open("debug.txt");
	int* h_result = (int*)malloc(sizeof(int));
	for (int i = 0; i < numBins; i++){
		reduceBin << <numBlocks, threadsPerBlock >> >(d_binMatrix, threadsPerBlock, d_histo, i, threadsPerBlock);
	}


	file.close();
	cudaFree(d_binMatrix);
	cudaFree(d_reduceBin_out);
	cudaFree(d_bucketValues);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
