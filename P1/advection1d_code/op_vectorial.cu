#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define BLOCKSIZE 1024
#define FLOAT_MIN 10
#define FLOAT_MAX 100

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

/**
 * @brief Desenrrollado de bucle del ultimo warp de cada bloque
 * @param sdata Puntero a memoria compartida de device
 * @param tid Identificador de hebra de GPU
 * @post sdata es modificado
*/
__device__ void warpReduce(volatile float *sdata, int tid)
{
    sdata[tid] = (sdata[tid] > sdata[tid + 32]) ? sdata[tid] : sdata[tid + 32];
    sdata[tid] = (sdata[tid] > sdata[tid + 16]) ? sdata[tid] : sdata[tid + 16];
    sdata[tid] = (sdata[tid] > sdata[tid + 8]) ? sdata[tid] : sdata[tid + 8];
    sdata[tid] = (sdata[tid] > sdata[tid + 4]) ? sdata[tid] : sdata[tid + 4];
    sdata[tid] = (sdata[tid] > sdata[tid + 2]) ? sdata[tid] : sdata[tid + 2];
    sdata[tid] = (sdata[tid] > sdata[tid + 1]) ? sdata[tid] : sdata[tid + 1];
}

/**
 * @brief Kernel para la reduccion
 * @param Min Vector a reducir
 * @param Mout Resultado de la reduccion
 * @note La reduccion se hace por bloques usando memoria compartida
 * por lo que el vector de salida no esta reducido completamente
*/
__global__ void reduce_max(float *Min, float *Mout, const int nverts)
{
    extern __shared__ float sdata[];

    // Cada hebra carga un elemento a memoria compartida
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Primera reduccion antes de cargar en memoria compartida
    sdata[tid] = (Min[i] > Min[i + blockDim.x]) ? Min[i] : Min[i + blockDim.x];
    __syncthreads();

    // Hacer reduccion en memoria compartida
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

/**
 * @brief Kernel que calcula el vector B del problema
 * @param B_in Vector con los datos de entrada
 * @param N Tamaño del vector
 * @note Esta version hace uso de memoria compartida
*/
__global__ void calcula_B_shared(float *B_in, int N)
{
    extern __shared__ float sdata[];

    float A_im2, A_im1, A_i, A_ip1, A_ip2;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        sdata[tid] = B_in[i];
        //__syncthreads(); // Esperar a que las hebras carguen en memoria compartida

        A_im2 = (i - 2 < 0) ? 0.0 : B_in[i - 2];
        A_im1 = (i - 1 < 0) ? 0.0 : B_in[i - 1];
        A_i = B_in[i];
        A_ip1 = (i + 1 > N) ? 0.0 : B_in[i + 1];
        A_ip2 = (i + 2 > N) ? 0.0 : B_in[i + 2];

        sdata[tid] = (pow(A_im2, 2) + 2 * pow(A_im1, 2) + pow(A_i, 2) - 3 * pow(A_ip1, 2) + 5 * pow(A_ip2, 2)) / 24.0;
    }

    // Copiar de memoria compartida a salida
    if (tid == 0)
    {
        int offset = blockIdx.x * blockDim.x;
        int posicion;

        for (int i = 0; i < blockDim.x; i++)
        {
            posicion = offset + i;

            if (posicion < N) // Necesario por las hebras que sobran
                B_in[posicion] = sdata[i];
        }
    }
}

/**
 * @brief Kernel que calcula el vector B del problema
 * @param B_in Vector con los datos de entrada
 * @param B_out Vector con los datos de salida
 * @param N Tamaño del vector
 * @note Esta version no hace uso de memoria compartida
*/
__global__ void calcula_B(float *B_in, float *B_out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float A_im2, A_im1, A_i, A_ip1, A_ip2;

    if (i < N)
    {
        A_im2 = (i - 2 < 0) ? 0.0 : B_in[i - 2];
        A_im1 = (i - 1 < 0) ? 0.0 : B_in[i - 1];
        A_i = B_in[i];
        A_ip1 = (i + 1 > N) ? 0.0 : B_in[i + 1];
        A_ip2 = (i + 2 > N) ? 0.0 : B_in[i + 2];

        B_out[i] = (pow(A_im2, 2) + 2 * pow(A_im1, 2) + pow(A_i, 2) - 3 * pow(A_ip1, 2) + 5 * pow(A_ip2, 2)) / 24.0;
    }
}

/**
 * @brief Genera número aleatorio
 * @note cambiar valores de macros 
 * para mayor o menor rango
*/
float generate_random_float()
{
    static default_random_engine generador;
    static uniform_real_distribution<float> distribucion_uniforme(FLOAT_MIN, FLOAT_MAX);
    return distribucion_uniforme(generador);
}

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

int main(int argc, char const *argv[])
{
    // Get GPU information
    int dev_id;
    int num_val;
    cudaDeviceProp props;

    GPU_ERR_CHK(cudaGetDevice(&dev_id));

    cudaGetDeviceProperties(&props, dev_id);

    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           dev_id, props.name, props.major, props.minor);

    cout << "Introduce numero de valores: ";
    cin >> num_val;

    //****************************
    // Inicializamos vector A
    //****************************
    float *A = new float[num_val]; //  Vector de entrada A

    for (int i = 0; i < num_val; i++)
        A[i] = generate_random_float();

    //****************************
    // Calculamos vector B en CPU
    //****************************
    float *B = new float[num_val];

    float A_im2, A_im1, A_i, A_ip1, A_ip2;

    double t_cpu_inicial = cpuSecond();

    for (int i = 0; i < num_val; i++)
    {
        A_im2 = (i - 2 < 0) ? 0.0 : A[i - 2];
        A_im1 = (i - 1 < 0) ? 0.0 : A[i - 1];
        A_i = A[i];
        A_ip1 = (i + 1 > num_val) ? 0.0 : A[i + 1];
        A_ip2 = (i + 2 > num_val) ? 0.0 : A[i + 2];
        B[i] = (pow(A_im2, 2) + 2 * pow(A_im1, 2) + pow(A_i, 2) - 3 * pow(A_ip1, 2) + 5 * pow(A_ip2, 2)) / 24.0;
    }

    double t_cpu_final = cpuSecond();
    double t_cpu = t_cpu_final - t_cpu_inicial;

    //*****************************************************
    // Calculamos vector B en GPU (sin memoria compartida)
    //*****************************************************
    float *d_A, *d_b;
    float *h_b = new float[num_val];

    GPU_ERR_CHK(cudaMalloc((void **)&d_A, num_val * sizeof(float)));
    GPU_ERR_CHK(cudaMalloc((void **)&d_b, num_val * sizeof(float)));
    GPU_ERR_CHK(cudaMemcpy(d_A, A, num_val * sizeof(float), cudaMemcpyHostToDevice));

    int blocks_per_grid = ceil((float)num_val / (float)BLOCKSIZE);

    double t_gpu_inicial_1 = cpuSecond();
    calcula_B<<<blocks_per_grid, BLOCKSIZE>>>(d_A, d_b, num_val);
    GPU_ERR_CHK(cudaDeviceSynchronize());
    double t_gpu_final_1 = cpuSecond();

    GPU_ERR_CHK(cudaGetLastError());
    GPU_ERR_CHK(cudaMemcpy(h_b, d_b, num_val * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_ERR_CHK(cudaDeviceSynchronize());

    double t_gpu_1 = t_gpu_final_1 - t_gpu_inicial_1;

    //*****************************************************
    // Calculamos vector B en GPU (con memoria compartida)
    //*****************************************************
    float *d_c;
    float *h_c = new float[num_val];
    int shared_mem_size = BLOCKSIZE * sizeof(float);

    GPU_ERR_CHK(cudaMalloc((void **)&d_c, num_val * sizeof(float)));
    GPU_ERR_CHK(cudaMemcpy(d_c, A, num_val * sizeof(float), cudaMemcpyHostToDevice));

    double t_gpu_inicial_2 = cpuSecond();
    calcula_B_shared<<<blocks_per_grid, BLOCKSIZE, shared_mem_size>>>(d_c, num_val);
    GPU_ERR_CHK(cudaDeviceSynchronize());
    double t_gpu_final_2 = cpuSecond();

    GPU_ERR_CHK(cudaGetLastError());
    GPU_ERR_CHK(cudaMemcpy(h_c, d_c, num_val * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_ERR_CHK(cudaDeviceSynchronize());

    double t_gpu_2 = t_gpu_final_2 - t_gpu_inicial_2;

    //******************
    // Reduccion en CPU
    //******************
    double t_red_cpu_ini = cpuSecond();

    float mayor_cpu = B[0];

    for (int i = 1; i < num_val; i++)
    {
        mayor_cpu = (B[i] > mayor_cpu) ? B[i] : mayor_cpu;
    }

    double t_red_cpu_fin = cpuSecond();
    double t_red_cpu = t_red_cpu_fin - t_red_cpu_ini;

    //******************
    // Reduccion en GPU
    //******************
    float *d_d, *d_e;                        // Parametros de entrada del kernel
    float *h_d = new float[blocks_per_grid]; // Salida del kernel en el host

    GPU_ERR_CHK(cudaMalloc((void **)&d_d, num_val * sizeof(float)));
    GPU_ERR_CHK(cudaMalloc((void **)&d_e, blocks_per_grid * sizeof(float)));
    GPU_ERR_CHK(cudaMemcpy(d_d, B, num_val * sizeof(float), cudaMemcpyHostToDevice));

    double t_gpu_inicial_3 = cpuSecond();
    reduce_max<<<blocks_per_grid, BLOCKSIZE, shared_mem_size>>>(d_d, d_e, num_val);
    GPU_ERR_CHK(cudaDeviceSynchronize());
    double t_gpu_final_3 = cpuSecond();

    GPU_ERR_CHK(cudaGetLastError());
    GPU_ERR_CHK(cudaMemcpy(h_d, d_e, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_ERR_CHK(cudaDeviceSynchronize());

    float mayor_gpu = h_d[0];

    for (int i = 1; i < blocks_per_grid; i++)
    {
        mayor_gpu = (h_d[i] > mayor_gpu) ? h_d[i] : mayor_gpu;
    }

    double t_red_gpu = t_gpu_final_3 - t_gpu_inicial_3;

    //****************************
    // Comprobacion CPU-GPU
    //****************************
    bool passed = true;
    int i = 0;

    while (passed && i < num_val)
    {
        if (B[i] != h_b[i] && B[i] != h_c[i])
        {
            cout << "ERR B[" << i << "] = " << B[i]
                 << " h_b[" << i << "] = " << h_b[i]
                 << " h_c[" << i << "] = " << h_c[i]
                 << endl;
            passed = false;
        }
        i++;
    }

    if (passed)
        cout << "PASSED TEST" << endl;
    else
        cout << "ERROR IN TEST" << endl;

    //********************
    // Mostrar resultados
    //********************
    cout << "Tiempo en CPU = " << t_cpu << endl
         << "Tiempo en GPU (sin memoria compartida) = " << t_gpu_1 << endl
         << "Tiempo en GPU (con memoria compartida) = " << t_gpu_2 << endl
         << "Ganancia (sin memoria compartida) = " << t_cpu / t_gpu_1 << endl
         << "Ganancia (con memoria compartida) = " << t_cpu / t_gpu_2 << endl
         << "Tiempo de reduccion en CPU = " << t_red_cpu << endl
         << "Tiempo de reduccion en GPU = " << t_red_gpu << endl
         << "Valor de reduccion en CPU = " << mayor_cpu << endl
         << "Valor de reduccion en GPU = " << mayor_gpu << endl;

    // Liberar memoria host
    delete (A);
    delete (B);
    delete (h_b);
    delete (h_c);
    delete (h_d);

    // Liberar memoria device
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);

    return 0;
}
