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

__global__ void checkBit(unsigned int* const d_inputVals, 
							unsigned int mask, 
							unsigned int value, 
							const size_t numElems, 
							unsigned int* d_vectorMask);

__global__ void exclusiveScan_Reduce(unsigned int* d_in,
							const size_t numElems, 
							unsigned int* d_exclusiveScanReduce_out);

__global__ void exclusiveScan_Reduce_PhaseOne(unsigned int* d_scanAllElements,
							const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void exclusiveScan_Reduce_PhaseTwo(const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void exclusiveScan_Downsweep_PhaseOne(const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void exclusiveScan_Downsweep_PhaseTwo(unsigned int* d_scanAllElements,
							const size_t elementsToProcess,
							unsigned int* d_scanIntermediateElements);

__global__ void switchPositions(unsigned int* d_scanResult1, 
							unsigned int* d_scanResult2, 
							unsigned int* const d_inputVals, 
							unsigned int* const d_inputPos,
							unsigned int* const d_outputVals, 
							unsigned int* const d_outputPos, 
							const size_t numElems);

#include <fstream>
#include <iostream>
using namespace std;

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	int nBits = 1; 
	int nBins = 1 << nBits;
	unsigned int mask;

	int threadsPerBlock = 1024;
	//int numberOfBlocks = numElems / threadsPerBlock + 1;
	int numberOfBlocks = 256;
	int totalThreads = threadsPerBlock * numberOfBlocks;

	unsigned int* d_vectorMask;
	checkCudaErrors(cudaMalloc(&d_vectorMask, totalThreads * sizeof(unsigned int)));

	ofstream myfile;
	myfile.open("debug.txt");
	mask = 1;
	checkBit << <numberOfBlocks, threadsPerBlock >> >(d_inputVals, mask, 0, numElems, d_vectorMask);

	unsigned int* d_scanAllElements_0;
	checkCudaErrors(cudaMalloc(&d_scanAllElements_0, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_scanAllElements_0, d_vectorMask, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	unsigned int* d_scanIntermediateElements_0;
	checkCudaErrors(cudaMalloc(&d_scanIntermediateElements_0, numberOfBlocks * sizeof(unsigned int)));

	exclusiveScan_Reduce_PhaseOne << < numberOfBlocks, threadsPerBlock >> > (d_scanAllElements_0, threadsPerBlock, d_scanIntermediateElements_0);
	exclusiveScan_Reduce_PhaseTwo << <1, numberOfBlocks >> >(numberOfBlocks, d_scanIntermediateElements_0);

	exclusiveScan_Downsweep_PhaseOne << <1, numberOfBlocks >> > (numberOfBlocks, d_scanIntermediateElements_0);
	exclusiveScan_Downsweep_PhaseTwo << <numberOfBlocks, threadsPerBlock >> >(d_scanAllElements_0, threadsPerBlock, d_scanIntermediateElements_0);

	checkBit << <numberOfBlocks, threadsPerBlock >> >(d_inputVals, mask, 1, numElems, d_vectorMask);

	/******************************************************************************/

	unsigned int* d_scanAllElements_1;
	checkCudaErrors(cudaMalloc(&d_scanAllElements_1, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpy(d_scanAllElements_1, d_vectorMask, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	unsigned int* d_scanIntermediateElements_1;
	checkCudaErrors(cudaMalloc(&d_scanIntermediateElements_1, numberOfBlocks * sizeof(unsigned int)));

	exclusiveScan_Reduce_PhaseOne << < numberOfBlocks, threadsPerBlock >> > (d_scanAllElements_1, threadsPerBlock, d_scanIntermediateElements_1);
	exclusiveScan_Reduce_PhaseTwo << <1, numberOfBlocks >> >(numberOfBlocks, d_scanIntermediateElements_1);

	exclusiveScan_Downsweep_PhaseOne << <1, numberOfBlocks >> > (numberOfBlocks, d_scanIntermediateElements_1);
	exclusiveScan_Downsweep_PhaseTwo << <numberOfBlocks, threadsPerBlock >> >(d_scanAllElements_1, threadsPerBlock, d_scanIntermediateElements_1);

	/******************************************************************************/

	unsigned int* h_exclusiveScanReduceResult_0 = (unsigned int*)malloc(totalThreads * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_exclusiveScanReduceResult_0, d_scanAllElements_0, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int* h_exclusiveScanReduceResult_1 = (unsigned int*)malloc(totalThreads * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_exclusiveScanReduceResult_1, d_scanAllElements_1, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < totalThreads; ++i)
	{
		myfile << h_exclusiveScanReduceResult_0[i] << h_exclusiveScanReduceResult_1[i];
		myfile << '\n';
	}



	/*for (int i = 0; i < 8 * sizeof(unsigned int); i += nBits){

		mask = (nBins - 1) << i;
		checkBit << <numberOfBlocks, threadsPerBlock >> >(d_inputVals, mask, 0, numElems, d_vectorMask);
		
		exclusiveScan_Reduce << < numberOfBlocks, threadsPerBlock >> > (d_vectorMask, threadsPerBlock, d_exclusive_scan_0_intermediate);
		exclusiveScan_Reduce << <1, numberOfBlocks >> >(d_exclusive_scan_0_intermediate, numberOfBlocks, d_exclusive_scan_0_final);


		//myfile << 
		


	}*/

	myfile.close();
	free(h_exclusiveScanReduceResult_0);
	free(h_exclusiveScanReduceResult_1);
	checkCudaErrors(cudaFree(d_vectorMask));
	checkCudaErrors(cudaFree(d_scanAllElements_0));
	checkCudaErrors(cudaFree(d_scanIntermediateElements_0));
	checkCudaErrors(cudaFree(d_scanAllElements_1));
	checkCudaErrors(cudaFree(d_scanIntermediateElements_1));
}


__global__ void checkBit(unsigned int* const d_inputVals,
	unsigned int mask,
	unsigned int value,
	const size_t numElems,
	unsigned int* d_vectorMask){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//CUIDADO AQUI: será mesmo maior ou igual? não será so maior?
	if (tid >= numElems){
		d_vectorMask[tid] = 0;
		return;
	}

	unsigned int inputAndMask = d_inputVals[tid] & mask;
	if (inputAndMask == value){
		d_vectorMask[tid] = 1;
	}
	else{
		d_vectorMask[tid] = 0;
	}

	__syncthreads();
	return;
}

__global__ void exclusiveScan_Reduce_PhaseOne(unsigned int* d_scanAllElements,
											const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements){

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;
	

	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_scanAllElements[tid] = d_scanAllElements[tid] + d_scanAllElements[tid - s];
		}
		__syncthreads();
	}

	if (threadX == 1023){
		d_scanIntermediateElements[blockIdx.x] = d_scanAllElements[tid];
	}
	__syncthreads();
	return;

}

__global__ void exclusiveScan_Reduce_PhaseTwo(const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements){

	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;

	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_scanIntermediateElements[tid] = d_scanIntermediateElements[tid] + d_scanIntermediateElements[tid - s];
		}
		__syncthreads();
	}
	return;

}



__global__ void exclusiveScan_Downsweep_PhaseOne(const size_t elementsToProcess,
											unsigned int* d_scanIntermediateElements){
	// cada thread que entrar aqui, seja na primeira ou na segunda passagem: o seu valor inicial tem que ser igual ao valor
	//que se encontra na sua posição no vector intermedio. 
	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;
	if (threadX == 255){
		d_scanIntermediateElements[255] = 0;
	}

	__syncthreads();
	//fase de downsweep
	int auxiliary;
	for (int s = elementsToProcess / 2, mod = elementsToProcess; s > 0; s = s / 2, mod = mod / 2){
		if ((threadX + 1) % mod == 0){
			auxiliary = d_scanIntermediateElements[tid - s];
			d_scanIntermediateElements[tid - s] = d_scanIntermediateElements[tid];
			d_scanIntermediateElements[tid] = d_scanIntermediateElements[tid] + auxiliary;
		}
		__syncthreads();
	}
	return;
}

__global__ void exclusiveScan_Downsweep_PhaseTwo(unsigned int* d_scanAllElements,
												const size_t elementsToProcess,
												unsigned int* d_scanIntermediateElements){
	int threadX = threadIdx.x;
	int tid = threadX + blockDim.x * blockIdx.x;
	if (threadX == 1023){
		d_scanAllElements[tid] = d_scanIntermediateElements[blockIdx.x];
	}

	__syncthreads();
	//fase de downsweep
	int auxiliary;
	for (int s = elementsToProcess / 2, mod = elementsToProcess; s > 0; s = s / 2, mod = mod / 2){
		if ((threadX + 1) % mod == 0){
			auxiliary = d_scanAllElements[tid - s];
			d_scanAllElements[tid - s] = d_scanAllElements[tid];
			d_scanAllElements[tid] = d_scanAllElements[tid] + auxiliary;
		}
		__syncthreads();
	}
	return;
}

__global__ void switchPositions(unsigned int* d_scanResult1,
	unsigned int* d_scanResult2,
	unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems){


}