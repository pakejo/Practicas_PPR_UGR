#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <math.h>

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

//**************************************************************************
__global__ void transformacion_no_shared(float *A, float *B, int N)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float Aim2, Aim1, Ai, Aip1, Aip2;

	if (i < N)
	{

		Aim2 = (i - 2 < 0) ? 0.0 : A[i - 2];
		Aim1 = (i - 1 < 0) ? 0.0 : A[i - 1];
		Aip1 = (i + 1 > N) ? 0.0 : A[i + 1];
		Aip2 = (i + 2 > N) ? 0.0 : A[i + 2];

		Ai = A[i];

		B[i] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) - 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0;
	}
}

//**************************************************************************
// Vector maximum  kernel

__global__ void reduceMax(float *V_in, float *V_out, const int N)
{
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : 0.0);
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)

			if (sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];

		__syncthreads();
	}
	if (tid == 0)
		V_out[blockIdx.x] = sdata[0];
}

/**************************************************************************
 **************************************************************************/
int main(int argc, char *argv[])
{

	int blockSize, N;

	if (argc != 3)
	{
		cerr << "Error en los argumentos: blockSize numValores" << endl;
		return (-1);
	}
	else
	{
		blockSize = atoi(argv[1]);
		//numBlocks = atoi(argv[2]);
		N = atoi(argv[2]);
	}

	//N = blockSize * numBlocks;

	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if (err != cudaSuccess)
	{
		cout << "ERRORRR" << endl;
	}

	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	cout << "Tamaño bloque: " << blockSize << endl;
	cout << "Nº valores: " << N << endl;

	//Variables
	int size = N * sizeof(float);
	float *A = new float[N];
	float *B = new float[N];

	float *A_CPU = new float[N];
	float *B_CPU = new float[N];
	int blocks_per_grid = ceil(float(N) / blockSize);
	float *B_out_red = new float[blocks_per_grid];

	float *A_device = NULL;
	float *B_device = NULL;
	float *B_d_red_in = NULL;
	float *B_d_red_out = NULL;

	//Initialize vector A (GPU) y A (CPU)
	for (int i = 0; i < N; i++)
	{
		A[i] = (float)(1 - (i % 100) * 0.001);
		A_CPU[i] = (float)(1 - (i % 100) * 0.001);
	}

	//Reserve memory
	err = cudaMalloc((void **)&A_device, size);
	if (err != cudaSuccess)
	{
		cout << "ERROR RESERVA [A Device]" << endl;
	}

	err = cudaMalloc((void **)&B_device, size);
	if (err != cudaSuccess)
	{
		cout << "ERROR RESERVA [B Device]" << endl;
	}

	err = cudaMalloc((void **)&B_d_red_in, size);
	if (err != cudaSuccess)
	{
		cout << "ERROR RESERVA [A Device Reduction INPUT]" << endl;
	}
	err = cudaMalloc((void **)&B_d_red_out, blocks_per_grid * sizeof(float));
	if (err != cudaSuccess)
	{
		cout << "ERROR RESERVA [A Device Reduction OUTPUT]" << endl;
	}

	/* ---------------------------------------------------------------------- */
	/* ------------------------------ CPU phase ----------------------------- */
	double t1 = cpuSecond();

	float Ai, Aim1, Aim2, Aip1, Aip2;
	float max = 0.0;

	for (int i = 0; i < N; i++)
	{

		Aim2 = (i - 2 < 0) ? 0.0 : A_CPU[i - 2];
		Aim1 = (i - 1 < 0) ? 0.0 : A_CPU[i - 1];
		Aip1 = (i + 1 > N) ? 0.0 : A_CPU[i + 1];
		Aip2 = (i + 2 > N) ? 0.0 : A_CPU[i + 2];

		Ai = A_CPU[i];

		B_CPU[i] = (pow(Aim2, 2) + 2.0 * pow(Aim1, 2) + pow(Ai, 2) - 3.0 * pow(Aip1, 2) + 5.0 * pow(Aip2, 2)) / 24.0;
	}
	double Tcpu_max = cpuSecond() - t1;

	for (int i = 0; i < N; i++)
	{
		max = (B_CPU[i] > max) ? B_CPU[i] : max;
	}

	cout << "Tiempo gastado CPU = " << Tcpu_max << endl;
	cout << "Máximo: " << max << endl;

	/* ---------------------------------------------------------------------- */
	/* ------------------ GPU phase >>[No shared memory]<< ------------------ */
	t1 = cpuSecond();

	//Host A to Device
	err = cudaMemcpy(A_device, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cout << "ERROR COPIA [A Device]" << endl;
	}

	int threadsPerBlock = blockSize;
	int blocksPerGrid = ceil((float)N / (float)threadsPerBlock);

	transformacion_no_shared<<<blocksPerGrid, threadsPerBlock>>>(A_device, B_device, N);

	//Device to Host
	cudaMemcpy(B, B_device, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//Time GPU
	double Tgpu = cpuSecond() - t1;
	cout << "Tiempo gastado GPU = " << Tgpu << endl
		 << endl;

	/* ------------------------------------------------------------------- */
	//							 GPU REDUCTION PHASE
	t1 = cpuSecond();

	//Host to device
	err = cudaMemcpy(B_d_red_in, B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		cout << "ERROR COPIA A GPU REDUCTION" << endl;
	}

	int shared_mem_size = threadsPerBlock * sizeof(float);
	reduceMax<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(B_d_red_in, B_d_red_out, N);
	cudaDeviceSynchronize();

	//Device to Host
	cudaMemcpy(B_out_red, B_d_red_out, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	max = 0.0;
	for (int i = 0; i < blocks_per_grid; i++)
		max = (B_out_red[i] > max) ? B_out_red[i] : max;

	//Time GPU Reduction
	double Tgpu_reduction = cpuSecond() - t1;
	cout << "Tiempo gastado GPU REDUCTION = " << Tgpu_reduction << endl;
	cout << "Máximo: " << max << endl
		 << endl;

	cout << "Ganancia [TGPU]= " << Tcpu_max / Tgpu << endl;
	cout << "Ganancia [TGPU reduction]= " << Tcpu_max / Tgpu_reduction << endl;
}
