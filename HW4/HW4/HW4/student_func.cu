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

__global__ void exclusiveScan_Downsweep(unsigned int* d_in,
										const size_t numElems,
										unsigned int* d_exclusiveScanDownsweep_out);

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

	unsigned int* d_exclusive_scan_0_reduce_final;
	unsigned int* d_exclusive_scan_1_reduce_final;
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_0_reduce_final, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_1_reduce_final, totalThreads * sizeof(unsigned int)));

	unsigned int* d_exclusive_scan_0_reduce_intermediate;
	unsigned int* d_exclusive_scan_1_reduce_intermediate;
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_0_reduce_intermediate, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_1_reduce_intermediate, totalThreads * sizeof(unsigned int)));

	unsigned int* d_exclusive_scan_0_downsweep_final;
	unsigned int* d_exclusive_scan_1_downsweep_final;
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_0_downsweep_final, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_1_downsweep_final, totalThreads * sizeof(unsigned int)));

	unsigned int* d_exclusive_scan_0_downsweep_intermediate;
	unsigned int* d_exclusive_scan_1_downsweep_intermediate;
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_0_downsweep_intermediate, totalThreads * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_exclusive_scan_1_downsweep_intermediate, totalThreads * sizeof(unsigned int)));
	
	ofstream myfile;
	myfile.open("debug.txt");
	mask = 1;
	checkBit << <numberOfBlocks, threadsPerBlock >> >(d_inputVals, mask, 0, numElems, d_vectorMask);
	exclusiveScan_Reduce << < numberOfBlocks, threadsPerBlock >> > (d_vectorMask, threadsPerBlock, d_exclusive_scan_0_intermediate);
	exclusiveScan_Reduce << <1, numberOfBlocks >> >(d_exclusive_scan_0_intermediate, numberOfBlocks, d_exclusive_scan_0_final);



	unsigned int* h_exclusiveScanReduceResult = (unsigned int*)malloc(totalThreads * sizeof(unsigned int));
	checkCudaErrors(cudaMemcpy(h_exclusiveScanReduceResult, d_exclusive_scan_0_reduce_final, totalThreads*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < totalThreads; ++i)
	{
		myfile << h_exclusiveScanReduceResult[i];
		myfile << '\n';
	}

	/*for (int i = 0; i < 8 * sizeof(unsigned int); i += nBits){

		mask = (nBins - 1) << i;
		checkBit << <numberOfBlocks, threadsPerBlock >> >(d_inputVals, mask, 0, numElems, d_vectorMask);
		
		exclusiveScan_Reduce << < numberOfBlocks, threadsPerBlock >> > (d_vectorMask, threadsPerBlock, d_exclusive_scan_0_intermediate);
		exclusiveScan_Reduce << <1, numberOfBlocks >> >(d_exclusive_scan_0_intermediate, numberOfBlocks, d_exclusive_scan_0_final);


		//myfile << 
		


	}*/

	//myfile.close();
	//free(h_exclusiveScanReduceResult);

	checkCudaErrors(cudaFree(d_vectorMask));
	checkCudaErrors(cudaFree(d_exclusive_scan_0_reduce_final));
	checkCudaErrors(cudaFree(d_exclusive_scan_1_reduce_final));
	checkCudaErrors(cudaFree(d_exclusive_scan_0_reduce_intermediate));
	checkCudaErrors(cudaFree(d_exclusive_scan_1_reduce_intermediate));
	checkCudaErrors(cudaFree(d_exclusive_scan_0_downsweep_final));
	checkCudaErrors(cudaFree(d_exclusive_scan_1_downsweep_final));
	checkCudaErrors(cudaFree(d_exclusive_scan_0_downsweep_intermediate));
	checkCudaErrors(cudaFree(d_exclusive_scan_1_downsweep_intermediate));
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

__global__ void exclusiveScan_Reduce(unsigned int* d_in,
	const size_t elementsToProcess,
	unsigned int* d_exclusiveScanReduce_out){

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int threadX = threadIdx.x;
	d_exclusiveScanReduce_out[tid] = d_in[tid];

	__syncthreads();

	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_exclusiveScanReduce_out[tid] = d_exclusiveScanReduce_out[tid] + d_exclusiveScanReduce_out[tid - s];
		}
		__syncthreads();
	}

	__syncthreads();
	return;

}

__global__ void exclusiveScan_Downsweep(unsigned int* d_in, const size_t numElems, unsigned int* d_exclusiveScanDownsweep_out){
	// cada thread que entrar aqui, seja na primeira ou na segunda passagem: o seu valor inicial tem que ser igual ao valor
	//que se encontra na sua posição no vector intermedio. 
}

__global__ void switchPositions(unsigned int* d_scanResult1,
	unsigned int* d_scanResult2,
	unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems){


}