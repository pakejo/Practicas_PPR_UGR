#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "Graph.h"

// Includes for Cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define blocksize 256
#define GPU_ERR_CHK(ans)                      \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

/**
 * @brief Comprueba el codigo de error de una llamada Cuda
 * @param code Codigo del error
 * @param file Archivo donde se produjo el error
 * @param line Linea que ha dado el error
 * @param abort Indica si debe abortar el programa ante el error. True por defecto
*/
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

using namespace std;

/**
 * @brief Calcula un instante de tiempo
 * @return Instante de tiempo
*/
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

/**
 * @brief Kernel que realiza el algoritmo de Floyd
 * @param M Matriz sobre la que ejecutar el algoritmo
 * @param nverts Numero de elementos de la matriz
 * @param k Iteracion del primer bucle
 * @post M es modificado tras la ejecucion
 * @note Version unidimensional
*/
__global__ void floyd_kernel(int *M, const int nverts, const int k)
{
	int ij = threadIdx.x + blockDim.x * blockIdx.x;

	if (ij < nverts * nverts)
	{
		int Mij = M[ij];
		int i = ij / nverts;
		int j = ij - i * nverts;

		if (i != j && i != k && j != k)
		{
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
	}
}

/**
 * @brief Kernel que realiza el algoritmo de Floyd
 * @param M Matriz sobre la que ejecutar el algoritmo
 * @param nverts Numero de elementos de la matriz
 * @param k Iteracion del primer bucle
 * @post M es modificado tras la ejecucion
 * @note Version bidimensional
*/
__global__ void floyd_kernel_bidimensional(int *M, const int nverts, const int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x; // fila
	int i = blockIdx.y * blockDim.y + threadIdx.y; // columna
	int index = i * nverts + j;

	if (i < nverts && j < nverts)
	{
		if (i != j && i != k && j != k) // No calcular la diagonal
		{
			int Aij = M[index];
			int Aikj = M[nverts * i + k] + M[nverts * k + j];

			M[index] = min(Aij, Aikj);
		}
	}
}

/**
 * @brief Desenrrollado de bucle del ultimo warp de cada bloque
 * @param sdata Puntero a memoria compartida de device
 * @param tid Identificador de hebra de GPU
 * @post sdata es modificado
*/
__device__ void warpReduce(volatile int *sdata, int tid)
{
	sdata[tid] = (sdata[tid] > sdata[tid + 32]) ? sdata[tid] : sdata[tid + 32];
	sdata[tid] = (sdata[tid] > sdata[tid + 16]) ? sdata[tid] : sdata[tid + 16];
	sdata[tid] = (sdata[tid] > sdata[tid + 8]) ? sdata[tid] : sdata[tid + 8];
	sdata[tid] = (sdata[tid] > sdata[tid + 4]) ? sdata[tid] : sdata[tid + 4];
	sdata[tid] = (sdata[tid] > sdata[tid + 2]) ? sdata[tid] : sdata[tid + 2];
	sdata[tid] = (sdata[tid] > sdata[tid + 1]) ? sdata[tid] : sdata[tid + 1];
}

/**
 * @brief Kernel que calcula el camino mas largo
 * @param Min Matriz con los datos de entrada
 * @param Mout Matriz con los datos de salida
 * @param nverts Numero de elementos de la matriz
*/
__global__ void compute_longest_path(int *Min, int *Mout, const int nverts)
{
	extern __shared__ int sdata[];

	// Cada hebra carga un elemento a memoria compartida
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Primera reduccion antes de cargar en memoria compartida
	sdata[tid] = (Min[i] > Min[i + blockDim.x]) ? Min[i] : Min[i + blockDim.x];
	__syncthreads();

	// hacer reduccion en memoria compartida
	for (int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
		__syncthreads();
	}

	if (tid < 32)
		warpReduce(sdata, tid);

	if (tid == 0)
		Mout[blockIdx.x] = sdata[0];
}


int main(int argc, char *argv[])
{

	if (argc != 2)
	{
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return (-1);
	}

	//Get GPU information
	int devID;
	cudaDeviceProp props;

	GPU_ERR_CHK(cudaGetDevice(&devID));

	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	// Variables del grafo
	Graph G;
	G.lee(argv[1]);
	const int nverts = G.vertices;
	const int niters = nverts;
	const int nverts2 = nverts * nverts;

	// Variables para el algoritmo
	int *c_Out_M_1D = new int[nverts2];
	int *c_Out_M_2D = new int[nverts2];
	int size = nverts2 * sizeof(int);
	int *d_In_M_1D = NULL;
	int *d_In_M_2D = NULL;

	// variables para la reduccion
	int nBlocksReduction = ceil(float(nverts2) / blocksize);
	int *h_Out_M_reduction = new int[nBlocksReduction];
	int *d_In_M_reduction = NULL;
	int *d_Out_M_reduction = NULL;

	// Reserva de memoria en device
	GPU_ERR_CHK(cudaMalloc((void **)&d_In_M_1D, size));

	GPU_ERR_CHK(cudaMalloc((void **)&d_In_M_2D, size));

	GPU_ERR_CHK(cudaMalloc((void **)&d_In_M_reduction, size));

	GPU_ERR_CHK(cudaMalloc((void **)&d_Out_M_reduction, nBlocksReduction * sizeof(int)));

	int *A = G.Get_Matrix();

	//***************************************************
	// Calculamos Floyd en GPU (version unidimiensional)
	//***************************************************
	double t1 = cpuSecond();

	GPU_ERR_CHK(cudaMemcpy(d_In_M_1D, A, size, cudaMemcpyHostToDevice));

	for (int k = 0; k < niters; k++)
	{
		int threadsPerBlock = blocksize;

		int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

		floyd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_In_M_1D, nverts, k);

		GPU_ERR_CHK(cudaGetLastError());
	}

	GPU_ERR_CHK(cudaMemcpy(c_Out_M_1D, d_In_M_1D, size, cudaMemcpyDeviceToHost));
	GPU_ERR_CHK(cudaDeviceSynchronize());

	double Tgpu_1D = cpuSecond() - t1;

	//**************************************************
	// Calculamos Floyd en GPU (version bidimiensional)
	//**************************************************
	t1 = cpuSecond();

	GPU_ERR_CHK(cudaMemcpy(d_In_M_2D, A, size, cudaMemcpyHostToDevice));

	for (int k = 0; k < niters; k++)
	{
		dim3 threadsPerBlock(16, 16);

		dim3 blocksPerGrid(ceil((float)(nverts) / threadsPerBlock.x), ceil((float)(nverts) / threadsPerBlock.y));

		floyd_kernel_bidimensional<<<blocksPerGrid, threadsPerBlock>>>(d_In_M_2D, nverts, k);

		GPU_ERR_CHK(cudaGetLastError());
	}

	GPU_ERR_CHK(cudaMemcpy(c_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaDeviceSynchronize());

	double Tgpu_2D = cpuSecond() - t1;

	//****************************
	// Calculamos Floyd en CPU
	//****************************
	t1 = cpuSecond();

	int inj, in, kn;
	for (int k = 0; k < niters; k++)
	{
		kn = k * nverts;
		for (int i = 0; i < nverts; i++)
		{
			in = i * nverts;
			for (int j = 0; j < nverts; j++)
				if (i != j && i != k && j != k)
				{
					inj = in + j;
					A[inj] = min(A[in + k] + A[kn + j], A[inj]);
				}
		}
	}

	double t2 = cpuSecond() - t1;

	//*****************************
	// Calculamos reduccion en GPU
	//*****************************
	t1 = cpuSecond();

	GPU_ERR_CHK(cudaMemcpy(d_In_M_reduction, c_Out_M_2D, size, cudaMemcpyHostToDevice));

	int gridSize = (nverts2 + blocksize - 1) / blocksize;

	int sharedMemSize = blocksize * sizeof(int);

	compute_longest_path<<<gridSize, blocksize, sharedMemSize>>>(d_In_M_reduction, d_Out_M_reduction, nverts2);

	GPU_ERR_CHK(cudaGetLastError());

	GPU_ERR_CHK(cudaMemcpy(h_Out_M_reduction, d_Out_M_reduction, nBlocksReduction * sizeof(int), cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaDeviceSynchronize());

	int mayor_gpu = h_Out_M_reduction[0];
	int val;

	for (int i = 1; i < nBlocksReduction; i++)
	{
		val = h_Out_M_reduction[i];
		mayor_gpu = (val > mayor_gpu) ? val : mayor_gpu;
	}

	double Tgpu_red = cpuSecond() - t1;

	//*****************************
	// Calculamos reduccion en CPU
	//*****************************

	t1 = cpuSecond();

	int mayor_cpu = c_Out_M_2D[0];
	for (int i = 0; i < nverts; i++)
		for (int j = 0; j < nverts; j++)
		{
			int val = c_Out_M_2D[i * nverts + j];
			mayor_cpu = (val > mayor_cpu) ? val : mayor_cpu;
		}
	double Tcpu_red = cpuSecond() - t1;

	//**********************
	// Comprobacion CPU-GPU
	//**********************
	for (int i = 0; i < nverts; i++)
		for (int j = 0; j < nverts; j++)
			if (abs(c_Out_M_1D[i * nverts + j] - G.arista(i, j)) > 0)
				cout << "1D[" << i * nverts + j << "]=" << c_Out_M_1D[i * nverts + j] << ", 2D[" << i * nverts + j << "]=" << c_Out_M_2D[i * nverts + j] << endl;

	for (int i = 0; i < nverts; i++)
		for (int j = 0; j < nverts; j++)
			if (abs(c_Out_M_2D[i * nverts + j] - G.arista(i, j)) > 0)
				cout << "2D (" << i << "," << j << ")   " << c_Out_M_2D[i * nverts + j] << "..." << G.arista(i, j) << endl;

	//********************
	// Mostrar resultados
	//********************
	cout << "Tiempo gastado Floyd CPU = " << t2 << endl
		 << "Tiempo gastado GPU (1D) = " << Tgpu_1D << endl
		 << "Tiempo gastado GPU (2D) = " << Tgpu_2D << endl
		 << "Ganancia Floyd 1D= " << t2 / Tgpu_1D << endl
		 << "Ganancia Floyd 2D= " << t2 / Tgpu_2D << endl
		 << "Ganancia Reduccion 1D = " << Tcpu_red / Tgpu_red << endl
		 << "Tiempo de reducción CPU = " << Tcpu_red << endl
		 << "Tiempo de reducción GPU = " << Tgpu_red << endl
		 << "Valor de reduccion en CPU = " << mayor_cpu << endl
		 << "Valor de reduccion en GPU = " << mayor_gpu << endl;

	// Liberar memoria host
	delete (c_Out_M_1D);
	delete (c_Out_M_2D);
	delete (h_Out_M_reduction);

	// Liberar memoria device
	cudaFree(d_In_M_1D);
	cudaFree(d_In_M_2D);
	cudaFree(d_In_M_reduction);
	cudaFree(d_Out_M_reduction);
}
