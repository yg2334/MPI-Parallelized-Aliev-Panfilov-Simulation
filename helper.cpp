/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"

#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

void printMat(const char mesg[], double *E, int m, int n);

extern control_block cb;


//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;
    int nprocs=1, myrank=0;
    #ifdef _MPI_
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank != 0){
      int nprocs=1, myrank=0;

        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        //  //printf("Process Rank: %d\n",myrank);
        int pgrid_mycol = myrank%cb.px;
        int pgrid_myrow = myrank/cb.px;
        int mesh_mycols = cb.n/cb.px;
        int mesh_myrows = cb.m/cb.py;
        if (cb.m%cb.py != 0){
        if (pgrid_myrow < cb.m%cb.py){
            mesh_myrows++;
        }
        }
        if (cb.n%cb.px != 0){
        if (pgrid_mycol < cb.n%cb.px){
            mesh_mycols++;
        }
        }
        m = mesh_myrows;
        n = mesh_mycols;
        for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }

    } else{
    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    }
    #else
    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }
    #endif

    // We only print the meshes if they are small enough
#if 1
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}


double *alloc1D(int m,int n){
    int nprocs=1, myrank=0;
    #ifdef _MPI
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank != 0){
      int nprocs=1, myrank=0;

        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        //  //printf("Process Rank: %d\n",myrank);
        int pgrid_mycol = myrank%cb.px;
        int pgrid_myrow = myrank/cb.px;
        int mesh_mycols = cb.n/cb.px;
        int mesh_myrows = cb.m/cb.py;
        if (cb.m%cb.py != 0){
        if (pgrid_myrow < cb.m%cb.py){
            mesh_myrows++;
        }
        }
        if (cb.n%cb.px != 0){
        if (pgrid_mycol < cb.n%cb.px){
            mesh_mycols++;
        }
        }
      int nx=mesh_myrows+2, ny=mesh_mycols+2;
      double *E;
      // Ensures that allocatdd memory is aligned on a 16 byte boundary
      assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
      return(E);  
    } else {
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
    }
    #else
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
    #endif

}

double *allocD(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
