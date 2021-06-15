#include <iostream>
#include <math.h>
#include <mpi.h>

using namespace std;

void imprime_matriz(int *matriz_A, int tam_fila)
{
    int cont = 0;

    for (int i = 0; i < tam_fila * tam_fila; i++)
    {
        cout << matriz_A[i] << "\t";
        cont++;

        if (cont == tam_fila)
        {
            cout << endl;
            cont = 0;
        }
    }
}

bool necesita_redondeo(int &tam_matriz_actual, int num_procesos)
{
    bool necesita_redondeo = false;

    if (tam_matriz_actual % num_procesos != 0) // No es multiplo, buscamos uno
    {
        int parte_entera = tam_matriz_actual / num_procesos;
        tam_matriz_actual = (parte_entera + 1) * num_procesos;
        necesita_redondeo = true;
    }

    return necesita_redondeo;
}

int main(int argc, char *argv[])
{
    double t_inicio, t_final;                       // Variables para medir el tiempo
    int rank, size;                                 // Variables necesarias para MPI al inicio
    int n;                                          // Tamaño de la matriz pasado por parametro
    int tam;                                        // Tamaño de cada submatriz
    int color_diagonal, color_fila, color_columna;  // Colores de los comunicadores
    int tam_matriz_nuevo;                           // Nuevo tamaño de la matriz (solo si hay que redondear)
    int rank_fila, rank_columna, rank_diagonal;     // Rangos de los procesos en los nuevos comunicadores
    int *matriz_A;                                  // Matriz a multiplicar
    int *buf_envio;                                 // Buffer para enviar la matriz a los procesos
    int *vector_X;                                  // Vector a multiplicar
    int *comprueba;                                 // Vector para comprobar resultados
    bool redondear;                                 // Comprueba si es necesario redondear
    MPI_Comm com_diagonal, com_filas, com_columnas; // Nuevos comunicadores

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int raiz_P = sqrt(size); // Raiz cuadrada del numero de procesadores

    n = atoi(argv[1]);
    redondear = necesita_redondeo(n, size);
    tam = n / raiz_P;

    int vector_X_local[tam];       // Porcion local a cada proceso del vector a multiplicar
    int buf_recep[n * n];          // Buffer para recibir cada submatriz
    int sumas_parciales[tam]{0};   // Vector con las sumas parciales de cada proceso
    int reduccion_fila[tam];       // Vector para reduccion de los resultados
    int vector_Y[raiz_P * raiz_P]; // Vector con los resultados finales

    if (rank == 0)
    {
        if (redondear)
        {
            tam_matriz_nuevo = n;
            n = atoi(argv[1]); // Recuperamos valor por defecto

            cout << n << " no es multiplo de " << size
                 << ". Redondeamos tamaño a " << tam_matriz_nuevo
                 << " y rellenamos con 0" << endl;

            matriz_A = new int[tam_matriz_nuevo * tam_matriz_nuevo];
            vector_X = new int[tam_matriz_nuevo];

            int limite_fila = n;
            int limite_columna = n;
            int cont_col_aux = 1;

            for (int i = 0; i < tam_matriz_nuevo * tam_matriz_nuevo; i++)
            {
                if (i % tam_matriz_nuevo == 0 && i != 0) // Llegamos al final de una fila
                {
                    limite_fila += tam_matriz_nuevo;
                    cont_col_aux++;
                }

                if (i < limite_fila)
                    matriz_A[i] = i;
                else
                    matriz_A[i] = 0;

                if (cont_col_aux >= n)
                    matriz_A[i] = 0;
            }

            for (int i = 0; i < tam_matriz_nuevo; i++)
            {
                vector_X[i] = (i < n) ? i : 0;
            }

            // Volvemos a poner n con valor correcto
            n = tam_matriz_nuevo;
        }
        else
        {
            cout << n << " es multiplo de " << size
                 << ". No es necesario redondear " << endl;

            matriz_A = new int[n * n];
            vector_X = new int[n];

            for (int i = 0; i < n * n; i++)
            {
                matriz_A[i] = i;
            }

            for (int i = 0; i < n; i++)
            {
                vector_X[i] = i;
            }
        }

        //imprime_matriz(matriz_A, n);

        /*Creo buffer de envío para almacenar los datos empaquetados*/
        buf_envio = new int[n * n];

        MPI_Datatype MPI_BLOQUE;
        MPI_Type_vector(tam, tam, n, MPI_INT, &MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);

        for (int i = 0, posicion = 0; i < size; i++)
        {
            int fila_P = i / raiz_P;
            int columna_P = i % raiz_P;
            int comienzo = (columna_P * tam) + (fila_P * tam * tam * raiz_P);
            MPI_Pack(&matriz_A[comienzo], 1, MPI_BLOQUE, buf_envio, sizeof(int) * n * n, &posicion, MPI_COMM_WORLD);
        }

        MPI_Type_free(&MPI_BLOQUE);

        // Calculamos producto matriz vector secuencial
        int tmp_index = 0;
        int tmp_index_2 = 0;
        int tam_sec = redondear ? tam_matriz_nuevo : n;
        comprueba = new int[tam_sec]{0};

        for (int i = 0; i < tam_sec * tam_sec; i++)
        {
            comprueba[tmp_index_2] += matriz_A[i] * vector_X[tmp_index];

            if (tmp_index == tam_sec - 1)
            {
                tmp_index = 0;
                tmp_index_2++;
            }
            else
                tmp_index++;
        }
    }

    // Scatter de la matriz. A partir de este punto cada proceso tiene su submatriz
    MPI_Scatter(buf_envio, sizeof(int) * tam * tam, MPI_PACKED, buf_recep, tam * tam, MPI_INT, 0, MPI_COMM_WORLD);

    // Matriz de coeficientes
    int matriz_ids[raiz_P][raiz_P];
    int cont = 0;

    for (int i = 0; i < raiz_P; i++)
    {
        cont = raiz_P * i;

        for (int j = 0; j < raiz_P; j++)
        {
            matriz_ids[i][j] = j + cont;
        }
    }

    // Crear comunicadores: 0: Diagonal, 1 a raiz_p: Filas, raiz_p + 1 a 2raiz_p: Columnas
    int color_diagonal_aux = MPI_UNDEFINED;
    int color_fila_aux = 1;
    int color_columna_aux = raiz_P + 1;

    /*Cada proceso calcula el color de todos, 
    pero solamente se quedara con los que le corresponden*/
    for (int i = 0; i < raiz_P; i++)
    {
        for (int j = 0; j < raiz_P; j++)
        {
            if (i == j)
                color_diagonal_aux = 0;
            else
                color_diagonal_aux = MPI_UNDEFINED;

            color_fila_aux = i + 1;
            color_columna_aux = raiz_P + j + 1;

            if (matriz_ids[i][j] == rank)
            {
                color_diagonal = color_diagonal_aux;
                color_fila = color_fila_aux;
                color_columna = color_columna_aux;
            }
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color_diagonal, rank, &com_diagonal);
    MPI_Comm_split(MPI_COMM_WORLD, color_fila, rank, &com_filas);
    MPI_Comm_split(MPI_COMM_WORLD, color_columna, rank, &com_columnas);

    MPI_Comm_rank(com_filas, &rank_fila);
    MPI_Comm_rank(com_columnas, &rank_columna);

    // Enviamos el vector_X a los procesos de la diagonal usando el comunicador creado
    if (color_diagonal == 0)
    {
        MPI_Scatter(&vector_X[0], tam, MPI_INT, &vector_X_local[0], tam, MPI_INT, 0, com_diagonal);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_inicio = MPI_Wtime();

    // Difundimos cada subvector pora la columna
    int raiz_broadcast = color_columna % (raiz_P + 1);
    MPI_Bcast(&vector_X_local[0], tam, MPI_INT, raiz_broadcast, com_columnas);

    // Multiplicacion matriz-vector
    int j = 0;
    int k = 0;

    for (int i = 0; i < tam * tam; i++)
    {
        sumas_parciales[k] += buf_recep[i] * vector_X_local[j];
        j++;

        if (j == tam)
        {
            j = 0;
            k++;
        }
    }

    // Reduccion por filas del vector_X_local
    int raiz_reduccion = color_fila - 1;
    MPI_Reduce(sumas_parciales, reduccion_fila, tam, MPI_INT, MPI_SUM, raiz_reduccion, com_filas);

    MPI_Barrier(MPI_COMM_WORLD);
    t_final = MPI_Wtime();

    if (color_diagonal == 0)
    {
        MPI_Gather(&reduccion_fila[0], tam, MPI_INT, &vector_Y[0], tam, MPI_INT, 0, com_diagonal);
    }

    // Comprobacion de resultados
    if (rank == 0)
    {
        int errores = 0;
        cout << "El resultado obtenido y el esperado son:" << endl;

        int tam_sec = redondear ? tam_matriz_nuevo : n;

        for (int i = 0; i < tam_sec; i++)
        {
            cout << "\t" << vector_Y[i] << "\t|\t" << comprueba[i] << endl;
            if (comprueba[i] != vector_Y[i])
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

        delete[] matriz_A;
        delete[] vector_X;
        delete[] buf_envio;
        delete[] comprueba;
    }

    MPI_Finalize();

    return 0;
}
