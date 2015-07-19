/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <fstream>
#include <iostream>
using namespace std;

__global__ void reduce_Minimum(float* d_in,
	int d_inSize,
	int numberOfBlocks,
	float* d_out);

__global__ void reduce_Maximum(float* d_in,
	int d_inSize,
	int numberOfBlocks,
	float* d_out);

__global__ void histogram_SeparateBuckets(const float* const d_in, int* d_threadBucketMatrix,
	int numberOfElements, int elementsPerThread, int pitch,
	float lumMin, float lumRange, int numBins);

__global__ void reduce_SumBuckets(int* d_in, int* d_out, int elementsToProcess);

__global__ void exclusiveScan(unsigned int* d_intermediate, int elementsToProcess);

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	int numberOfPixels = numRows*numCols;
	int threadsPerBlock = 1024;
	int numberOfBlocks = (numberOfPixels / threadsPerBlock) + 1;

	float* d_blockMinimum;
	float* d_blockMaximum;
	float* d_in;
	checkCudaErrors(cudaMalloc(&d_in, numberOfPixels*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_in, d_logLuminance, numberOfPixels*sizeof(float), cudaMemcpyDeviceToDevice));

	
	checkCudaErrors(cudaMalloc(&d_blockMinimum, numberOfBlocks*sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_blockMaximum, numberOfBlocks*sizeof(float)));

	reduce_Minimum << <numberOfBlocks, threadsPerBlock >> >(d_in,
															numberOfPixels, 
															numberOfBlocks, 
															d_blockMinimum);

	reduce_Maximum << <numberOfBlocks, threadsPerBlock >> >(d_in,
															numberOfPixels,
															numberOfBlocks,
															d_blockMaximum);

	reduce_Minimum << <1, threadsPerBlock >> >(d_blockMinimum,
															numberOfBlocks,
															1,
															d_blockMinimum);

	reduce_Maximum << <1, threadsPerBlock >> >(d_blockMaximum,
															numberOfBlocks,
															1,
															d_blockMaximum);

	float* h_blockMinimum = (float*)malloc(numberOfBlocks*sizeof(float));
	checkCudaErrors(cudaMemcpy(h_blockMinimum, d_blockMinimum, numberOfBlocks*sizeof(float), cudaMemcpyDeviceToHost));
	float* h_blockMaximum = (float*)malloc(numberOfBlocks*sizeof(float));
	checkCudaErrors(cudaMemcpy(h_blockMaximum, d_blockMaximum, numberOfBlocks*sizeof(float), cudaMemcpyDeviceToHost));
	min_logLum = h_blockMinimum[0];
	max_logLum = h_blockMaximum[0];
	checkCudaErrors(cudaFree(d_blockMinimum));
	checkCudaErrors(cudaFree(d_blockMaximum));
	
	free(h_blockMaximum);
	free(h_blockMinimum);

	int* d_threadBucketMatrix;
	size_t pitch;
	int pixelsPerThread = numberOfPixels / threadsPerBlock + 1;
	int width = threadsPerBlock*sizeof(int);
	int length = numBins*sizeof(int);
	
	checkCudaErrors(cudaMallocPitch(&d_threadBucketMatrix, &pitch, width, length));
	float lumRange = max_logLum - min_logLum;

	////////quantos pixels foram calculados pela thread
	/*int *d_test;
	int somatest = 0;*//*
	checkCudaErrors(cudaMalloc(&d_test, threadsPerBlock*sizeof(int)));*/

	histogram_SeparateBuckets << <1, threadsPerBlock >> >(d_logLuminance, d_threadBucketMatrix, numberOfPixels,
		pixelsPerThread, pitch, min_logLum, lumRange, numBins);

	/*int* h_test = (int*)malloc(threadsPerBlock * sizeof(int));
	checkCudaErrors(cudaMemcpy(h_test, d_test, threadsPerBlock*sizeof(int), cudaMemcpyDeviceToHost));
	ofstream testout("anothertest.txt");
	
	for (int k = 0; k < threadsPerBlock; k++){
		testout << h_test[k];
		somatest+=h_test[k];
	}*/


	int** h_threadBucketMatrix = (int**)malloc(numBins*sizeof(int*));
	for (int k = 0; k < numBins; k++){
		h_threadBucketMatrix[k] = (int*)malloc(threadsPerBlock*sizeof(int));
	}
	for (int k = 0; k < numBins; k++){
		checkCudaErrors(cudaMemcpy(h_threadBucketMatrix[k], (int*)((char*)d_threadBucketMatrix + k * pitch), threadsPerBlock*sizeof(int), cudaMemcpyDeviceToHost));
	}

	
	/*int summ = 0;
	if (fout.is_open()){
		for (int i = 0; i < numBins; i++){
			for (int k = 0; k < threadsPerBlock; k++){
				fout << (h_threadBucketMatrix[i][k]);
				summ += h_threadBucketMatrix[i][k];
			}
			fout << "\n";
		}
	}*/

	int* h_buckets = (int*)malloc(numBins*sizeof(int));
	int* d_out;
	int* d_bucketValues;
	checkCudaErrors(cudaMalloc(&d_bucketValues, threadsPerBlock*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(int)));
	
	for (int k = 0; k < numBins; k++){
		checkCudaErrors(cudaMemcpy(d_bucketValues, h_threadBucketMatrix[k], threadsPerBlock*sizeof(int), cudaMemcpyHostToDevice));
		//o numero de elementos a processar nesta funçao devia ser o numero de buckets,
		// e nao o numero de threads, que por coincidencia é igual. refactorizar. 
		reduce_SumBuckets << <1, threadsPerBlock >> > (d_bucketValues, d_out, threadsPerBlock);
		checkCudaErrors(cudaMemcpy(&h_buckets[k], d_out, sizeof(int), cudaMemcpyDeviceToHost));
	
	}

	
	//int sum = 0;
	//if (fout.is_open()){
	//	for (int i = 0; i < numBins; i++){
	//		fout << h_buckets[i];
	//		fout << '-';
	//		sum += h_buckets[i];
	//	}
	//	fout << "\n"; fout << sum;
	//}

	//fazer scan exclusivo de h_buckets

	unsigned int * d_intermediate;
	checkCudaErrors(cudaMalloc(&d_intermediate, numBins*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_intermediate, h_buckets, numBins*sizeof(int), cudaMemcpyHostToDevice));
	/*int *d_test;
	checkCudaErrors(cudaMalloc(&d_test, sizeof(int)));*/
	exclusiveScan << <1, threadsPerBlock >> >(d_intermediate, numBins);
	/*int *h_test = (int*)malloc(sizeof(int));
	checkCudaErrors(cudaMemcpy(h_test, d_test, sizeof(int), cudaMemcpyDeviceToHost));*/
	checkCudaErrors(cudaMemcpy(d_cdf, d_intermediate, numBins * sizeof(int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(h_buckets, d_intermediate, numBins*sizeof(int), cudaMemcpyDeviceToHost));

	ofstream fout("text.txt");
	for (int k = 0; k < numBins; k++){
		fout << h_buckets[k];
		fout << '-';
	}
	fout << '\n';

//	int test = *h_test;
//	free(h_test);
	free(h_buckets);
	checkCudaErrors(cudaFree(d_bucketValues));
	checkCudaErrors(cudaFree(d_in));
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_threadBucketMatrix));
	for (int i = 0; i < numBins; i++){
		free(h_threadBucketMatrix[i]);
	}
	free(h_threadBucketMatrix);/*
	checkCudaErrors(cudaFree(d_test));*/
	checkCudaErrors(cudaFree(d_intermediate));
	return;


}


__global__ void exclusiveScan(unsigned int* d_intermediate, int elementsToProcess){
	
	int threadX = threadIdx.x;
	int tid = threadX + blockIdx.x * blockDim.x;
	if (tid > elementsToProcess){
		return;
	}
	
	
	//fase de reduce. dintermediate tem que entrar aqui ja igual a d_in. 
	for (int s = 1, mod = 2; s <= elementsToProcess / 2; s = s * 2, mod = mod * 2){
		if ((threadX + 1) % mod == 0){
			d_intermediate[tid] = d_intermediate[tid] + d_intermediate[tid - s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 1023){
		d_intermediate[1023] = 0;
	}

	__syncthreads();
	//fase de downsweep
	int auxiliary;
	for (int s = elementsToProcess / 2, mod = elementsToProcess; s > 0; s = s / 2, mod = mod / 2){
		if ((threadX + 1) % mod == 0){
			auxiliary = d_intermediate[tid - s];
			d_intermediate[tid - s] = d_intermediate[tid];
			d_intermediate[tid] = d_intermediate[tid] + auxiliary;
		}
		__syncthreads();
	}

	//if (threadIdx.x == 1023){
	//	*d_out = d_intermediate[1023];
	//}
	
}

//Esta função dá asneira se o numero de buckets (o numero de elementos a processar) for maior do que
//o numero de threads definidas por bloco (1024). refactorizar. 
__global__ void reduce_SumBuckets(int* d_in, int* d_out, int elementsToProcess){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid > elementsToProcess){
		return;	
	}

	for (int s = elementsToProcess/2; s > 0; s = s / 2){
		if (threadIdx.x < s){
			d_in[tid] = d_in[tid] + d_in[tid + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0){
		*d_out = d_in[0];
	}

}

__global__ void histogram_SeparateBuckets(const float* const d_in, int* d_threadBucketMatrix, 
											int numberOfElements, int elementsPerThread, int pitch,
											float lumMin, float lumRange, int numBins){
	

	int pixelToRead;
	int threadX = threadIdx.x;
	int* gridAddress;

	for (int i = 0; i < numBins; i++){
		gridAddress = (int*)((char*)d_threadBucketMatrix + i * pitch) + threadX;
		*gridAddress = 0;
	}
	unsigned int bucket;
	float pixelValue;
//	int test = 0;
	for (int i = 0; i < elementsPerThread; i++){
		pixelToRead = threadX * elementsPerThread + i;
		if (pixelToRead >= numberOfElements){
			//d_test[threadX] = i;
			return;
		}
		pixelValue = d_in[pixelToRead];
		//bucket = floor((((float)(pixelValue - lumMin) / lumRange) * numBins));
		/////
		bucket = fminf(((unsigned int)(numBins - 1)), (unsigned int)((pixelValue - lumMin) / lumRange * numBins));

		/*if (bucket > 1023){
			bucket = 1023;
		}*/
		gridAddress = (int*)((char*)d_threadBucketMatrix + bucket * pitch) + threadX;
		*gridAddress = *gridAddress + 1;
		//test = i;
	}
	//d_test[threadX] = test;
	return;
}



__global__ void reduce_Minimum(float* d_in,
	int d_inSize,
	int numberOfBlocks,
	float* d_out)
{
	int threadsPerBlock = blockDim.x;
	int thisBlockId = blockIdx.x;
	int threadIndex = threadIdx.x;
	int vectorX = thisBlockId * threadsPerBlock + threadIndex;
	if (vectorX > d_inSize){
		return;
	}

	int elementsToProcess;
	if (thisBlockId + 1 == numberOfBlocks){
		elementsToProcess = d_inSize % threadsPerBlock;
	}
	else{
		elementsToProcess = threadsPerBlock;
	}

	int odd = 0;
	for (int s = elementsToProcess / 2; s > 0; s = s / 2){
		if (threadIndex < s){
			odd = elementsToProcess % 2;
			d_in[vectorX] = fminf(d_in[vectorX], d_in[vectorX + s + odd]);
			elementsToProcess -= s;
			if (s % 2 && elementsToProcess == 2){
				s = 2;
			}

		}
		__syncthreads();
	}

	if (threadIndex == 0){
		d_out[thisBlockId] = d_in[vectorX];
	}

	return;
}


__global__ void reduce_Maximum(float* d_in,
	int d_inSize,
	int numberOfBlocks,
	float* d_out)
{
	int threadsPerBlock = blockDim.x;
	int thisBlockId = blockIdx.x;
	int threadIndex = threadIdx.x;
	int vectorX = thisBlockId * threadsPerBlock + threadIndex;
	if (vectorX > d_inSize){
		return;
	}
	
	int elementsToProcess;
	if (thisBlockId + 1 == numberOfBlocks){
		elementsToProcess = d_inSize % threadsPerBlock;
	}
	else{
		elementsToProcess = threadsPerBlock;
	}

	int odd = 0;
	for (int s = elementsToProcess / 2; s > 0; s = s/2){
		if (threadIndex < s){
			odd = elementsToProcess % 2;
			d_in[vectorX] = fmaxf(d_in[vectorX], d_in[vectorX + s + odd]);
			elementsToProcess -= s;
			if (s % 2 && elementsToProcess == 2){
				s = 2;
			}
			
		}
		__syncthreads();
	}

	if (threadIndex == 0){
		d_out[thisBlockId] = d_in[vectorX];
	}

	return;
}


