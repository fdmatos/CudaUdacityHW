//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#include <fstream>
#include <iostream>
using namespace std;

__global__ void checkBit(unsigned int* const d_inputVals, 
							unsigned int mask, 
							unsigned int value, 
							const size_t numElems, 
							unsigned int* d_vectorMask);

__global__ void exclusiveScan_Reduce_PhaseOne(unsigned int* d_scanAllElements,
							const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void exclusiveScan_Reduce_PhaseTwo(const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements,
							int* d_scanReduceSum);

__global__ void exclusiveScan_Downsweep_PhaseOne(const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void exclusiveScan_Downsweep_PhaseTwo(unsigned int* d_scanAllElements,
							const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void switchPositions(unsigned int* d_scanResult0, 
							unsigned int* d_scanResult1, 
							unsigned int* const d_inputVals, 
							unsigned int* const d_inputPos,
							unsigned int* const d_outputVals, 
							unsigned int* const d_outputPos, 
							const size_t numElems,
							unsigned int mask,
							int* firstOnePosition);

void printInputVals(){

}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	int nBits = 1; 
	int nBins = 1 << nBits;
	

	int threadsPerBlock = 1024;
	int numberOfBlocks = 256;
	int totalThreads = threadsPerBlock * numberOfBlocks;

	unsigned int mask;
	mask = 1;

	unsigned int* d_scanAllElements_0;
	checkCudaErrors(cudaMalloc(&d_scanAllElements_0, totalThreads * sizeof(unsigned int)));
	unsigned int* d_scanIntermediateElements_0;
	checkCudaErrors(cudaMalloc(&d_scanIntermediateElements_0, numberOfBlocks * sizeof(unsigned int)));

	unsigned int* d_scanAllElements_1;
	checkCudaErrors(cudaMalloc(&d_scanAllElements_1, totalThreads * sizeof(unsigned int)));
	unsigned int* d_scanIntermediateElements_1;
	checkCudaErrors(cudaMalloc(&d_scanIntermediateElements_1, numberOfBlocks * sizeof(unsigned int)));

	unsigned int *d_vals_src = d_inputVals;
	unsigned int *d_pos_src = d_inputPos;
	unsigned int *d_vals_dst = d_outputVals;
	unsigned int *d_pos_dst = d_outputPos;

	unsigned int* d_vectorMask;
	checkCudaErrors(cudaMalloc(&d_vectorMask, totalThreads * sizeof(unsigned int)));


	int* d_scanReduceSum;
	checkCudaErrors(cudaMalloc(&d_scanReduceSum, sizeof(int)));
	int* h_scanReduceSum = (int*)malloc(sizeof(int));
	for (int i = 0; i < 8 * sizeof(unsigned int); i += nBits){
		mask = (nBins - 1) << i;

		/******************************************************************************/

		/*Continuo sem perceber por que é que as vezes, ao fazer as coisas em memoria partilhada, da erros*/
		checkBit << <numberOfBlocks, threadsPerBlock, threadsPerBlock>> >(d_vals_src, mask, 1 << i, numElems, d_vectorMask);
		
		checkCudaErrors(cudaMemcpy(d_scanAllElements_1, d_vectorMask, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

		exclusiveScan_Reduce_PhaseOne << < numberOfBlocks, threadsPerBlock>> > (d_scanAllElements_1, threadsPerBlock, d_scanIntermediateElements_1);
		exclusiveScan_Reduce_PhaseTwo << <1, numberOfBlocks >> >(numberOfBlocks, d_scanIntermediateElements_1, d_scanReduceSum);

		exclusiveScan_Downsweep_PhaseOne << <1, numberOfBlocks >> > (numberOfBlocks, d_scanIntermediateElements_1);
		exclusiveScan_Downsweep_PhaseTwo << <numberOfBlocks, threadsPerBlock >> >(d_scanAllElements_1, threadsPerBlock, d_scanIntermediateElements_1);


		/******************************************************************************/

		checkBit << <numberOfBlocks, threadsPerBlock >> >(d_vals_src, mask, 0, numElems, d_vectorMask);

		checkCudaErrors(cudaMemcpy(d_scanAllElements_0, d_vectorMask, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

		exclusiveScan_Reduce_PhaseOne << < numberOfBlocks, threadsPerBlock, threadsPerBlock*sizeof(unsigned int) >> > (d_scanAllElements_0, threadsPerBlock, d_scanIntermediateElements_0);
		exclusiveScan_Reduce_PhaseTwo << <1, numberOfBlocks >> >(numberOfBlocks, d_scanIntermediateElements_0, d_scanReduceSum);
		
		exclusiveScan_Downsweep_PhaseOne << <1, numberOfBlocks >> > (numberOfBlocks, d_scanIntermediateElements_0);
		exclusiveScan_Downsweep_PhaseTwo << <numberOfBlocks, threadsPerBlock >> >(d_scanAllElements_0, threadsPerBlock, d_scanIntermediateElements_0);

		switchPositions << <numberOfBlocks, threadsPerBlock >> >(d_scanAllElements_0, d_scanAllElements_1, d_vals_src, d_pos_src, d_vals_dst, d_pos_dst, numElems, mask, d_scanReduceSum);

		std::swap(d_vals_src, d_vals_dst);
		std::swap(d_pos_src, d_pos_dst);
	}
	checkCudaErrors(cudaMemcpy(d_outputVals, d_vals_src, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_pos_src, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaFree(d_vectorMask));
	checkCudaErrors(cudaFree(d_scanAllElements_0));
	checkCudaErrors(cudaFree(d_scanIntermediateElements_0));
	checkCudaErrors(cudaFree(d_scanAllElements_1));
	checkCudaErrors(cudaFree(d_scanIntermediateElements_1));
	checkCudaErrors(cudaFree(d_scanReduceSum));
	free(h_scanReduceSum);
}

//A memoria partilhada aqui não faz grande diferença, visto que o acesso a memoria global so seria feito uma vez. Mas mal não faz.
__global__ void checkBit(unsigned int* const d_inputVals,
	unsigned int mask,
	unsigned int value,
	const size_t numElems,
	unsigned int* d_vectorMask){

	__shared__ unsigned int d_inputVals_local[1024];

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;

	if (tid >= numElems){
		d_vectorMask[tid] = 0;
		return;
	}

	d_inputVals_local[threadX] = d_inputVals[tid];
	__syncthreads();

	unsigned int inputAndMask = d_inputVals_local[threadX] & mask;
	if (inputAndMask == value){
		d_vectorMask[tid] = 1;
	}
	else{
		d_vectorMask[tid] = 0;
	}
	
	return;
}


/* Nas quatro funçoes que se seguem a memoria partilhada é bastante util. O tempo de execução com memoria partilhada nestas quatro kernels seguintes
(vs memoria global nas quatro kernels) foi de 50 ms vs 90 ms (caiu para metade). Por alguma razao, nao consegui alocar memoria partilhada dinamicamente
(utilizando extern e especificando a quantidade na chamada à kernel. Já tinha acontecido no hw3. */
__global__ void exclusiveScan_Reduce_PhaseOne(unsigned int* d_scanAllElements,
											const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements){

	__shared__ unsigned int d_scanAllElements_local[1024];

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;
	
	d_scanAllElements_local[threadX] = d_scanAllElements[tid];
	__syncthreads();

	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_scanAllElements_local[threadX] = d_scanAllElements_local[threadX] + d_scanAllElements_local[threadX - s];
		}
		__syncthreads();
	}

	if (threadX == 1023){
		d_scanIntermediateElements[blockIdx.x] = d_scanAllElements_local[threadX];
	}

	d_scanAllElements[tid] = d_scanAllElements_local[threadX];
	__syncthreads();

	return;


}

__global__ void exclusiveScan_Reduce_PhaseTwo(const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements,
											int* d_scanReduceSum){

	__shared__ unsigned int d_scanIntermediateElements_local[256];

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;

	d_scanIntermediateElements_local[threadX] = d_scanIntermediateElements[tid];
	__syncthreads();

	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_scanIntermediateElements_local[tid] = d_scanIntermediateElements_local[tid] + d_scanIntermediateElements_local[tid - s];
		}
		__syncthreads();
	}

	if (threadX == 255){
		*d_scanReduceSum = d_scanIntermediateElements_local[255];
	}

	d_scanIntermediateElements[tid] = d_scanIntermediateElements_local[threadX];
	__syncthreads();

	return;

}



__global__ void exclusiveScan_Downsweep_PhaseOne(const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements){

	__shared__ unsigned int d_scanIntermediateElements_local[256];

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;

	d_scanIntermediateElements_local[threadX] = d_scanIntermediateElements[tid];
	if (threadX == 255){
		d_scanIntermediateElements_local[255] = 0;
	}
	__syncthreads();

	//fase de downsweep
	int auxiliary;
	for (int s = elementsToProcess / 2, mod = elementsToProcess; s > 0; s = s / 2, mod = mod / 2){
		if ((threadX + 1) % mod == 0){
			auxiliary = d_scanIntermediateElements_local[tid - s];
			d_scanIntermediateElements_local[tid - s] = d_scanIntermediateElements_local[tid];
			d_scanIntermediateElements_local[tid] = d_scanIntermediateElements_local[tid] + auxiliary;
		}
		__syncthreads();
	}

	d_scanIntermediateElements[tid] = d_scanIntermediateElements_local[threadX];
	__syncthreads();

	return;
}

__global__ void exclusiveScan_Downsweep_PhaseTwo(unsigned int* d_scanAllElements,
												const size_t elementsToProcess,
												unsigned int* d_scanIntermediateElements){

	__shared__ unsigned int d_scanAllElements_local[1024];

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;

	d_scanAllElements_local[threadX] = d_scanAllElements[tid];
	
	if (threadX == 1023){
		d_scanAllElements_local[threadX] = d_scanIntermediateElements[blockIdx.x];
	}
	__syncthreads();

	//fase de downsweep
	int auxiliary;
	for (int s = elementsToProcess / 2, mod = elementsToProcess; s > 0; s = s / 2, mod = mod / 2){
		if ((threadX + 1) % mod == 0){
			auxiliary = d_scanAllElements_local[threadX - s];
			d_scanAllElements_local[threadX - s] = d_scanAllElements_local[threadX];
			d_scanAllElements_local[threadX] = d_scanAllElements_local[threadX] + auxiliary;
		}
		__syncthreads();
	}

	d_scanAllElements[tid] = d_scanAllElements_local[threadX];
	__syncthreads();

	return;
}

//*Duvido que aqui sirva de muito usar memoria partilhada, uma vez que o acesso a memoria global so é feita duas vezes.
__global__ void switchPositions(unsigned int* d_scanResult0,
	unsigned int* d_scanResult1,
	unsigned int* d_inputVals,
	unsigned int* d_inputPos,
	unsigned int* d_outputVals,
	unsigned int* d_outputPos,
	const size_t numElems,
	unsigned int mask,
	int* firstOnePosition){

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;
	int total = numElems - 1;
	if (tid >= numElems){
		return;
	}

	if ((d_inputVals[tid] & mask) == 0){
		d_outputVals[d_scanResult0[tid]] = d_inputVals[tid];
		d_outputPos[d_scanResult0[tid]] = d_inputPos[tid];
	}
	else{
		d_outputVals[*firstOnePosition + d_scanResult1[tid]] = d_inputVals[tid];
		d_outputPos[*firstOnePosition + d_scanResult1[tid]] = d_inputPos[tid];
	}
	return;
}