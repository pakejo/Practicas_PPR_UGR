#include <iostream>
#include <random>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[])
{
    int size;               // Numero total de procesos
    int rank;               // ID del proceso
    int n_filas;            // Numero de filas de la matriz
    int tam_fila;           // Tamaño de una fila
    int *resultado;         // Resultado de cada multiplicacion
    int *matriz;            // Matriz a multiplicar
    int *vector;            // Vector a multiplicar
    int *vector_parcial;    // Vector para guardar resultado scatter
    int *vector_resultados; // Vector con todos los resultados reunidos con gather
    int *comprueba;         // Resultado secuencial, debe ser igual a vector_resultados
    double t_inicio;        // Tiempo en el que comienza la ejecucion
    double t_final;         // Tiempo en el que termina la ejecucion

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3)
    {
        cerr << "Uso: ./P2 numero_filas tamaño_fila" << endl;
        return -1;
    }

    n_filas = atoi(argv[1]);
    tam_fila = atoi(argv[2]);

    vector = new int[tam_fila];

    if (rank == 0)
    {
        matriz = new int[n_filas * tam_fila];
        comprueba = new int[n_filas];
        vector_resultados = new int[n_filas];

        // Llenar la matriz
        for (int i = 0; i < n_filas * tam_fila; i++)
        {
            matriz[i] = i; //random_int();
        }

        // Llenar vector
        for (int i = 0; i < tam_fila; i++)
        {
            vector[i] = i; //random_int();
        }

        // Inicializar vector de comprobacion
        for (int i = 0; i < n_filas; i++)
        {
            comprueba[i] = 0; //random_int();
        }

        // Calculamos producto matriz vector secuencial
        int tmp_index = 0;
        int tmp_index_2 = 0;

        for (int i = 0; i < n_filas * tam_fila; i++)
        {
            comprueba[tmp_index_2] += matriz[i] * vector[tmp_index];

            if (tmp_index == tam_fila - 1)
            {
                tmp_index = 0;
                tmp_index_2++;
            }
            else
                tmp_index++;
        }
    }

    // Calculamos cuantas filas le corresponde a cada proceso
    int n_filas_proceso = n_filas / size;
    int n_filas_restantes = n_filas % size;

    if (rank == size - 1)
        vector_parcial = new int[(n_filas_proceso + n_filas_restantes) * tam_fila];
    else
        vector_parcial = new int[n_filas_proceso * tam_fila];

    // Repartimos las filas entre los procesos
    int sendcounts[size]; // Numero de elementos que vamos a enviar a cada proceso
    int displ[size];      // Desplazamiento de cada bloque
    int recvcounts[size]; // Numero de elementos que recibimos de cada proceso

    for (int i = 0; i < size; i++)
    {
        displ[i] = i * tam_fila * n_filas_proceso;

        if (i != size - 1)
        {
            sendcounts[i] = n_filas_proceso * tam_fila;
            recvcounts[i] = n_filas_proceso;
        }
        else
        {
            sendcounts[i] = (n_filas_proceso + n_filas_restantes) * tam_fila;
            recvcounts[i] = n_filas_proceso + n_filas_restantes;
        }
    }

    // Hacemos scatter de la matriz
    MPI_Scatterv(matriz, sendcounts, displ, MPI_INT, vector_parcial, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Hacemos broadcast del vector
    MPI_Bcast(vector, tam_fila, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_inicio = MPI_Wtime();

    int j = 0;
    int k = 0;
    int n_resultados = recvcounts[rank];
    resultado = new int[n_resultados]{0};

    for (int i = 0; i < sendcounts[rank]; i++)
    {
        resultado[k] += vector_parcial[i] * vector[j];
        j++;

        if (j == tam_fila)
        {
            j = 0;
            k++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_final = MPI_Wtime();

    for (int i = 0; i < size; i++)
    {
        displ[i] = i * n_filas_proceso;
    }

    MPI_Gatherv(&resultado[0], n_resultados, MPI_INT, vector_resultados, recvcounts, displ, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0)
    {
        int errores = 0;
        cout << "El resultado obtenido y el esperado son:" << endl;
        for (unsigned int i = 0; i < n_filas; i++)
        {
            cout << "\t" << vector_resultados[i] << "\t|\t" << comprueba[i] << endl;
            if (comprueba[i] != vector_resultados[i])
                errores++;
        }

        if (errores)
        {
            cout << "Hubo " << errores << " errores." << endl;
        }
        else
        {
            cout << "No hubo errores" << endl;
        }
        cout << "El tiempo tardado ha sido " << t_final - t_inicio << " segundos." << endl;
    }

    return 0;
}