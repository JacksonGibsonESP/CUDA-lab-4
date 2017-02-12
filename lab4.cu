#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

using namespace std;

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
			    }														\
} while (0)

__device__ __constant__ double avg_const[96];

__global__ void kernel_main(int height, int width, int nc, unsigned int *src)
{
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
	while (tid_y < height)
	{
		int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
		while (tid_x < width)
		{
			double p[3];
			double max = 0;
			int jc = 0;
			p[0] = src[tid_y * width + tid_x] & 0xFF;
			p[1] = (src[tid_y * width + tid_x] & 0xFF00) >> 8;
			p[2] = (src[tid_y * width + tid_x] & 0xFF0000) >> 16;
			for (int k = 0; k < nc; k++)
			{
				double arg = p[0] * avg_const[k * 3] + p[1] * avg_const[k * 3 + 1] + p[2] * avg_const[k * 3 + 2];
				if (arg > max)
				{
					max = arg;
					jc = k;
				}
			}
			src[tid_y * width + tid_x] |= jc << 24;
			tid_x += blockDim.x * gridDim.x;
		}
		tid_y += blockDim.y * gridDim.y;
	}
}

int main()
{
	string path_in, path_out;

	cin >> path_in >> path_out;

	int nc;
	cin >> nc;

	int width, height;
	FILE *in = fopen(path_in.c_str(), "rb");
	if (in == NULL)
	{
		cout << "ERROR: Incorrect input file.\n";
		return 0;
	}
	fread(&width, sizeof(int), 1, in);
	fread(&height, sizeof(int), 1, in);

	if (width <= 0 || height <= 0 || nc < 0 || nc > 32)
	{
		cout << "ERROR: Incorrect data.\n";
		return 0;
	}

	unsigned int *src = (unsigned int *)malloc(sizeof(unsigned int) * width * height);
	fread(src, sizeof(unsigned int), width * height, in);
	fclose(in);

	double avg[96] = { 0, };

	for (int i = 0; i < nc; i++)
	{
		double np;
		cin >> np;
		for (int j = 0; j < np; j++)
		{
			int x, y;
			cin >> x >> y;
			avg[i * 3]		+= src[y * width + x] & 0xFF;
			avg[i * 3 + 1]	+= (src[y * width + x] & 0xFF00) >> 8;
			avg[i * 3 + 2]	+= (src[y * width + x] & 0xFF0000) >> 16;
		}
		avg[i * 3] /= np;
		avg[i * 3 + 1] /= np;
		avg[i * 3 + 2] /= np;
		double modulus = sqrt(avg[i * 3] * avg[i * 3] + avg[i * 3 + 1] * avg[i * 3 + 1] + avg[i * 3 + 2] * avg[i * 3 + 2]);
		avg[i * 3] /= modulus;
		avg[i * 3 + 1] /= modulus;
		avg[i * 3 + 2] /= modulus;
	}

	CSC(cudaMemcpyToSymbol(avg_const, avg, sizeof(double) * 96));

	unsigned int *src_dev;
	CSC(cudaMalloc(&src_dev, sizeof(unsigned int) * height * width));
	CSC(cudaMemcpy(src_dev, src, sizeof(unsigned int) * height * width, cudaMemcpyHostToDevice));

	dim3 threads_count(16, 16);
	dim3 blocks_count(16, 16);

	kernel_main << < blocks_count, threads_count >> >(height, width, nc, src_dev);

	CSC(cudaMemcpy(src, src_dev, sizeof(unsigned int) * height * width, cudaMemcpyDeviceToHost));
	CSC(cudaFree(src_dev));

	FILE *out = fopen(path_out.c_str(), "wb");
	if (out == NULL)
	{
		cout << "ERROR: Incorrect output file.\n";
		return 0;
	}
	fwrite(&width, sizeof(int), 1, out);
	fwrite(&height, sizeof(int), 1, out);
	fwrite(src, sizeof(unsigned int), height * width, out);
	fclose(out);

	free(src);
	return 0;
}
