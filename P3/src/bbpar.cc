#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

//using namespace std;

unsigned int NCIUDADES;
int rank, size, anterior, siguiente;
bool token_presente;
MPI_Comm comunicadorCarga, comunicadorCota;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    switch (argc)
    {
    case 3:
        NCIUDADES = atoi(argv[1]);
        break;
    default:
        std::cerr << "La sintaxis es: bbpar <tamaño> <archivo>" << std::endl;
        exit(1);
        break;
    }

    int **tsp0;           // Matriz de adyacencia
    int iteraciones;      // Numero de iteracion
    int U;                // Valor de la cota superior
    int color_comm_carga; // Color del comunicador para balanceo de carga
    int color_comm_cota;  // Color para el comunicador de difusion de cota
    bool nueva_U;         // Hay nueva cota superior
    bool fin;             // Condicion de fin
    tNodo nodo;           // Nodo a explorar
    tNodo lnodo;          // Hijo izquierdo
    tNodo rnodo;          // Hijo derecho
    tNodo solucion;       // Nodo solucion
    tPila pila;           // Pila de nodos a explorar
    double t_ini;         // Incio de la medicion de tiempo
    double t_fin;         // Fin de la medicion de tiempo

    color_comm_carga = 1;
    color_comm_cota = 2;

    MPI_Comm_split(MPI_COMM_WORLD, color_comm_carga, rank, &comunicadorCarga);
    MPI_Comm_split(MPI_COMM_WORLD, color_comm_cota, rank, &comunicadorCota);

    siguiente = (rank + 1) % size;
    anterior = (rank - 1 + size) % size;
    iteraciones = 0;
    U = INFINITO;
    fin = false;
    tsp0 = reservarMatrizCuadrada(NCIUDADES);
    InicNodo(&nodo);

    if (rank == 0)
    {
        LeerMatriz(argv[2], tsp0);
        token_presente = true;
    }

    // Difundir matriz al resto de procesos
    MPI_Bcast(&tsp0[0][0], NCIUDADES * NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_ini = MPI_Wtime();

    // Pedir al proceso 0 nodos para trabajar
    if (rank != 0)
    {
        token_presente = false;

        EquilibrioCarga(&pila, &fin, &solucion);

        if (!fin)
            pila.pop(nodo);
    }

    // Ciclo Branch & Bound
    while (!fin)
    { // ciclo del Branch&Bound
        Ramifica(&nodo, &lnodo, &rnodo, tsp0);
        nueva_U = false;
        if (Solucion(&rnodo))
        {
            if (rnodo.ci() < U)
            { // se ha encontrado una solucion mejor
                U = rnodo.ci();
                nueva_U = true;
                CopiaNodo(&rnodo, &solucion);
            }
        }
        else
        { //  no es un nodo solucion
            if (rnodo.ci() < U)
            { //  cota inferior menor que cota superior
                if (!pila.push(rnodo))
                {
                    printf("Error: pila agotada (%i)\n", rank);
                    liberarMatriz(tsp0);
                    exit(1);
                }
            }
        }
        if (Solucion(&lnodo))
        {
            if (lnodo.ci() < U)
            { // se ha encontrado una solucion mejor
                U = lnodo.ci();
                nueva_U = true;
                CopiaNodo(&lnodo, &solucion);
            }
        }
        else
        { // no es nodo solucion
            if (lnodo.ci() < U)
            { // cota inferior menor que cota superior
                if (!pila.push(lnodo))
                {
                    printf("Error: pila agotada (%i)\n", rank);
                    liberarMatriz(tsp0);
                    exit(1);
                }
            }
        }

#ifdef DIFUSION_COTA
        DifusionCotaSuperior(U, nueva_U);
#endif
        if (nueva_U)
        {
            pila.acotar(U);
        }

        EquilibrioCarga(&pila, &fin, &solucion);

        if (!fin)
            pila.pop(nodo);

        iteraciones++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_fin = MPI_Wtime();

    std::cout << "Proceso " << rank << ". Numero de iteraciones totales: " << iteraciones << std::endl;

    if (rank == 0)
    {
        std::cout << "Mejor solucion encontrada: ";
        EscribeNodo(&solucion);
        std::cout << "Encontrada en " << t_fin - t_ini << " segundos." << std::endl;
    }

    liberarMatriz(tsp0);
    MPI_Finalize();
    return 0;
}
