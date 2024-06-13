/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */
 
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

double *alloc1D(int m,int n);
double *allocD(int m,int n);

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 //printf("Solve Function\n");
 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n , x =cb.px , y =cb.py;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
 
 #ifdef _MPI_
 int nprocs=1, myrank=0;

 MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
 MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
//  //printf("Process Rank: %d\n",myrank);
 int pgrid_mycol = myrank%x;
 int pgrid_myrow = myrank/x;
 int mesh_mycols = n/x;
 int mesh_myrows = m/y;
 if (m%y != 0){
  if (pgrid_myrow < m%y){
     mesh_myrows++;
  }
 }
 if (n%x != 0){
  if (pgrid_mycol < n%x){
     mesh_mycols++;
  }
 }

 int mytot_pts = (mesh_myrows+2)*(mesh_mycols+2);
 MPI_Request reqs[8];
 MPI_Request req[2*(nprocs)];

 MPI_Datatype ghost_column;
 MPI_Type_vector(mesh_myrows,1,mesh_mycols+2, MPI_DOUBLE, &ghost_column);
 MPI_Type_commit(&ghost_column);
 
 MPI_Datatype ghost_row;
 MPI_Type_contiguous(mesh_mycols, MPI_DOUBLE, &ghost_row);
 MPI_Type_commit(&ghost_row);
 
 double *my_E, *my_R, *my_E_prev;
 
 my_E = allocD(mesh_myrows+2,mesh_mycols+2);
 my_E_prev = allocD(mesh_myrows+2,mesh_mycols+2);
 my_R = allocD(mesh_myrows+2,mesh_mycols+2);

 int my_startcol = pgrid_mycol * mesh_mycols;

 if (pgrid_mycol >= n%x) {
    my_startcol = my_startcol + n%x;
 } 
 
 int my_startrow = pgrid_myrow * mesh_myrows;

 if (pgrid_myrow >= m%y) {
    my_startrow = my_startrow + m%y;
 } 
 
 int count = 0;
 if (myrank==0){

    for (int proc =1 ; proc < nprocs; proc++){
        int pgrid_col = proc%x;
        int pgrid_row = proc/x;
        int mesh_cols = n/x;
        int mesh_rows = m/y;
        if (m%y != 0){
         if (pgrid_row < m%y){
            mesh_rows++;
         }
        }
        if (n%x != 0){
         if (pgrid_col < n%x){
            mesh_cols++;
         }
        }
        //printf("Proc Rank: %d\n",proc);
        int tot_pts = (mesh_rows+2)*(mesh_cols+2);
        //printf("total pts: %d\n",tot_pts);
        double *proc_R, *proc_E_prev;
        
        proc_E_prev = allocD(mesh_rows+2,mesh_cols+2);
        proc_R = allocD(mesh_rows+2,mesh_cols+2);

        int proc_startcol = pgrid_col * mesh_cols;
         
        if (pgrid_col >= n%x) {
            proc_startcol = proc_startcol + n%x;
        } 
    
        int proc_startrow = pgrid_row * mesh_rows;

        if (pgrid_myrow >= m%y) {
            proc_startrow = proc_startrow + m%y;
        }
	
        for(int j = 1; j < mesh_rows+1; j++){
            for(int i =1; i < mesh_cols+1; i++){
                proc_E_prev[(j)*(mesh_cols+2)+(i)] = E_prev[(proc_startrow+j)*(m+2)+ (proc_startcol+i)];
                proc_R[(j)*(mesh_cols+2)+(i)] = R[(proc_startrow+j)*(m+2)+ (proc_startcol+i)];
            }
        }
        //printf("MPI SEND\n");
        // MPI_Isend(proc_E_prev,tot_pts,MPI_DOUBLE,proc,0,MPI_COMM_WORLD,&req[count++]);
        // MPI_Isend(proc_R,tot_pts,MPI_DOUBLE,proc,1,MPI_COMM_WORLD,&req[count++]);
        MPI_Send(proc_E_prev,tot_pts,MPI_DOUBLE,proc,0,MPI_COMM_WORLD);
        MPI_Send(proc_R,tot_pts,MPI_DOUBLE,proc,1,MPI_COMM_WORLD);
        free (proc_E_prev);
        free (proc_R);

    }
    
    for(int j = 1; j < mesh_myrows+1; j++){
            for(int i =1; i < mesh_mycols+1; i++){
                my_E_prev[(j)*(mesh_mycols+2)+(i)] = E_prev[(my_startrow+j)*(m+2)+ (my_startcol+i)];
                my_R[(j)*(mesh_mycols+2)+(i)] = R[(my_startrow+j)*(m+2)+ (my_startcol+i)];
            }
        }
 } else {
    //printf("MPI RECV\n");
    // MPI_Irecv(my_E_prev, mytot_pts, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,&req[count++]);
    // MPI_Irecv(my_R, mytot_pts, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,&req[count++]);
    MPI_Recv(my_E_prev, mytot_pts, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Recv(my_R, mytot_pts, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
 }

//  MPI_Waitall(count,req,MPI_STATUSES_IGNORE);


 #endif


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    #ifdef _MPI_
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j,req_count = 0;

    // double *mycolumnArray1 = new double[mesh_myrows];
    // double *mycolumnArray2 = new double[mesh_myrows];
    // double *mycolumnArray3 = new double[mesh_myrows];
    // double *mycolumnArray4 = new double[mesh_myrows];

    // Fills in the TOP Ghost Cells
    if(pgrid_myrow == 0){
     for (i = 0; i < (mesh_mycols+2); i++) {
        my_E_prev[i] = my_E_prev[i + (mesh_mycols+2)*2];
     }
    }

    // Fills in the RIGHT Ghost Cells
    if(pgrid_mycol == x-1){
     for (i = (mesh_mycols+1); i < (mesh_myrows+2)*(mesh_mycols+2); i+=(mesh_mycols+2)) {
        my_E_prev[i] = my_E_prev[i-2];
     }
    }

    // Fills in the LEFT Ghost Cells
    if(pgrid_mycol == 0){
     for (i = 0; i < (mesh_myrows+2)*(mesh_mycols+2); i+=(mesh_mycols+2)) {
        my_E_prev[i] = my_E_prev[i+2];
     }	
    }

    // Fills in the BOTTOM Ghost Cells
    if(pgrid_myrow == y-1){
     for (i = ((mesh_myrows+2)*(mesh_mycols+2)-(mesh_mycols+2)); i < (mesh_mycols+2)*(mesh_myrows+2); i++) {
        my_E_prev[i] = my_E_prev[i - (mesh_mycols+2)*2];
     }
    }
    if(!cb.noComm){
    
    if (y != 1){
    	if (pgrid_myrow == 0){
    	  MPI_Isend(my_E_prev+((mesh_myrows)*(mesh_mycols+2)+1),1,ghost_row,myrank+x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+((mesh_myrows+1)*(mesh_mycols+2)+1),1,ghost_row,myrank+x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	} else if (pgrid_myrow == y-1) {
    	  MPI_Isend(my_E_prev+(mesh_mycols+3),1,ghost_row,myrank-x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+1,1,ghost_row,myrank-x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	} else {
    	  MPI_Isend(my_E_prev+(mesh_mycols+3),1,ghost_row,myrank-x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+1,1,ghost_row,myrank-x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Isend(my_E_prev+((mesh_myrows)*(mesh_mycols+2)+1),1,ghost_row,myrank+x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+((mesh_myrows+1)*(mesh_mycols+2)+1),1,ghost_row,myrank+x,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	}
    }
    
    if (x != 1){
    	if (pgrid_mycol == 0){
    // 	  int count = 0;
 	//   //printf("Loading");
	//   for(int k = (2*mesh_mycols+2); k <= (mesh_myrows)*(mesh_mycols+2)+mesh_mycols;k+=mesh_mycols+2){
	//  	mycolumnArray1[count] = my_E_prev[k];
	//  	count++;
	//   }
    	  MPI_Isend(my_E_prev+(2*mesh_mycols+2),1,ghost_column,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+(2*mesh_mycols+3),1,ghost_column,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	//   MPI_Isend(mycolumnArray1, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	//   MPI_Irecv(mycolumnArray2, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  //MPI_Isend(mycolumnArray1, mesh_myrows, MPI_DOUBLE, myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
 	  //MPI_Irecv(mycolumnArray2, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	} else if (pgrid_mycol == x-1) {
    	  //printf("Sending");
	//   int count = 0;
	//   for(int k = (mesh_mycols+3); k <= (mesh_myrows)*(mesh_mycols+2)+1;k+=mesh_mycols+2){
	//  	mycolumnArray3[count] = my_E_prev[k];
	//   	count++;
	//   }
    	  MPI_Irecv(my_E_prev+(mesh_mycols+2),1,ghost_column,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
          MPI_Isend(my_E_prev+(mesh_mycols+3),1,ghost_column,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	//   MPI_Isend(mycolumnArray3, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	//   MPI_Irecv(mycolumnArray4, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  //MPI_Isend(mycolumnArray3, mesh_myrows, MPI_DOUBLE, myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
 	  //MPI_Irecv(mycolumnArray4, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	} else {
    // 	  int count = 0;
 	//   //printf("Loading");
	//   for(int k = (2*mesh_mycols+2); k <= (mesh_myrows)*(mesh_mycols+2)+mesh_mycols;k+=mesh_mycols+2){
	//  	mycolumnArray1[count] = my_E_prev[k];
	//  	count++;
	//   }
	  
	//   int count1 = 0;
	//   for(int k = (mesh_mycols+3); k <= (mesh_myrows)*(mesh_mycols+2)+1;k+=mesh_mycols+2){
	//  	mycolumnArray3[count1] = my_E_prev[k];
	//   	count1++;
	//   }
	  /*
	  MPI_Isend(mycolumnArray1, mesh_myrows, MPI_DOUBLE, myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
	  MPI_Irecv(mycolumnArray2, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
	  MPI_Isend(mycolumnArray3, mesh_myrows, MPI_DOUBLE, myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
	  MPI_Irecv(mycolumnArray4, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
 	  */
        /*
 	      MPI_Isend(mycolumnArray1, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(mycolumnArray2, mesh_myrows, MPI_DOUBLE,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Isend(mycolumnArray3, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(mycolumnArray4, mesh_myrows, MPI_DOUBLE,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
          */
 	  
    	  MPI_Isend(my_E_prev+(2*mesh_mycols+2),1,ghost_column,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);
    	  MPI_Irecv(my_E_prev+(mesh_mycols+2),1,ghost_column,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);    	  
          MPI_Isend(my_E_prev+(mesh_mycols+3),1,ghost_column,myrank-1,0,MPI_COMM_WORLD, &reqs[req_count++]);
          MPI_Irecv(my_E_prev+(2*mesh_mycols+3),1,ghost_column,myrank+1,0,MPI_COMM_WORLD, &reqs[req_count++]);

    
    	}
    } 
    
    innerBlockRowStartIndex = 2*(mesh_mycols+2)+1;
    innerBlockRowEndIndex = (((mesh_myrows+1)*(mesh_mycols+2) - 1) - (mesh_mycols))-(mesh_mycols+2);
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex+1; j <= innerBlockRowEndIndex+1; j+=(mesh_mycols+2)) {
    E_tmp = my_E + j;
    E_prev_tmp = my_E_prev + j;
    R_tmp = my_R + j;
    for(i = 0; i < mesh_mycols-2; i++) {
    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
    E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
    R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
    }
    }
    
    
    MPI_Waitall(req_count,reqs,MPI_STATUSES_IGNORE);
    /*
    if(x!=1){
	 if (pgrid_mycol == 0){
		 int count = 0;
		 //printf("Recieved");
		 for(int k = (2*mesh_mycols+3); k <= (mesh_myrows)*(mesh_mycols+2)+mesh_mycols+1;k+=mesh_mycols+2){
			 my_E_prev[k] = mycolumnArray2[count];
			 count++;
	 	 }
	 	//printf("Unloading");
	 } else if (pgrid_mycol == x-1) {
		 //printf("Recieved");
		 int count = 0;
		 for(int k = (mesh_mycols+2); k <= (mesh_myrows)*(mesh_mycols+2);k+=mesh_mycols+2){
			 my_E_prev[k] = mycolumnArray4[count];
			 count++;
	 	 }
	 	//printf("Unloading");
	 } else {
		 //printf("Recieved");
		 int count = 0;
		 for(int k = (2*mesh_mycols+3); k <= (mesh_myrows)*(mesh_mycols+2)+mesh_mycols+1;k+=mesh_mycols+2){
			 my_E_prev[k] = mycolumnArray2[count];
			 count++;
		 }
		 int count1 = 0;
		 for(int k = (mesh_mycols+2); k <= (mesh_myrows)*(mesh_mycols+2);k+=mesh_mycols+2){
			 my_E_prev[k] = mycolumnArray4[count1];
			 count1++;
		 }
		 //printf("Unloading");
	 }*/
    
    } /*else {
        
        // Fills in the TOP Ghost Cells
    
     for (i = 0; i < (mesh_mycols+2); i++) {
        my_E_prev[i] = my_E_prev[i + (mesh_mycols+2)*2];
     }
    

    // Fills in the RIGHT Ghost Cells
    
     for (i = (mesh_mycols+1); i < (mesh_myrows+2)*(mesh_mycols+2); i+=(mesh_mycols+2)) {
        my_E_prev[i] = my_E_prev[i-2];
     }
    

    // Fills in the LEFT Ghost Cells
    
     for (i = 0; i < (mesh_myrows+2)*(mesh_mycols+2); i+=(mesh_mycols+2)) {
        my_E_prev[i] = my_E_prev[i+2];
     }	
    

    // Fills in the BOTTOM Ghost Cells
    
     for (i = ((mesh_myrows+2)*(mesh_mycols+2)-(mesh_mycols+2)); i < (mesh_mycols+2)*(mesh_myrows+2); i++) {
        my_E_prev[i] = my_E_prev[i - (mesh_mycols+2)*2];
     }
    
    }*/
    #else
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

    #endif

//////////////////////////////////////////////////////////////////////////////
//printf("Ending Program 1 \n");
#define FUSED 1
#ifdef FUSED
#ifdef _MPI_
    if (cb.noComm){
    innerBlockRowStartIndex = (mesh_mycols+2)+1;
    innerBlockRowEndIndex = (((mesh_myrows+2)*(mesh_mycols+2) - 1) - (mesh_mycols)) - (mesh_mycols+2);
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(mesh_mycols+2)) {
        E_tmp = my_E + j;
	E_prev_tmp = my_E_prev + j;
        R_tmp = my_R + j;
	for(i = 0; i < mesh_mycols; i++) {
	        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
    } 
    else {
     innerBlockRowStartIndex = (mesh_mycols+2)+1;
     innerBlockRowEndIndex = (((mesh_myrows+2)*(mesh_mycols+2) - 1) - (mesh_mycols)) - (mesh_mycols+2);
     // Solve for the excitation, a PDE

     j = innerBlockRowStartIndex;
     //j = mesh_mycols+3 ;
     E_tmp = my_E + j;
     E_prev_tmp = my_E_prev + j;
     R_tmp = my_R + j;
     for(int i = 0; i < mesh_mycols; i++) {
     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
     }

     j = innerBlockRowEndIndex;
     //j = ((mesh_myrows)*(mesh_mycols+2)+1);
     E_tmp = my_E + j;
     E_prev_tmp = my_E_prev + j;
     R_tmp = my_R + j;
     for(int i = 0; i < mesh_mycols; i++) {
     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
     }

     for(int j = innerBlockRowStartIndex+(mesh_mycols+2); j <= innerBlockRowEndIndex-(mesh_mycols+2); j+=(mesh_mycols+2)) {
     E_tmp = my_E + j;
     E_prev_tmp = my_E_prev + j;
     R_tmp = my_R + j;
     for(int i = 0; i < mesh_mycols; i+=mesh_mycols-1) {
     E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
     E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
     R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
     }
     }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif

#else

#ifdef _MPI_
    // Solve for the excitation, a PDE
    innerBlockRowStartIndex = (mesh_mycols+2)+1;
    innerBlockRowEndIndex = (((mesh_mycols+2)*(mesh_myrows+2) - 1) - (mesh_mycols)) - (mesh_mycols+2);
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(mesh_mycols+2)) {
            E_tmp = my_E + j;
            E_prev_tmp = my_E_prev + j;
            for(i = 0; i < mesh_mycols; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(mesh_mycols+2)]+E_prev_tmp[i-(mesh_mycols+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(mesh_mycols+2)) {
        E_tmp = my_E + j;
        R_tmp = my_R + j;
	    E_prev_tmp = my_E_prev + j;
        for(i = 0; i < mesh_mycols; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	    E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
#endif
//printf("Ending Program 2 \n");
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes

   #ifdef _MPI_
   double *tmp = my_E; my_E = my_E_prev; my_E_prev = tmp;
   #else
   double *tmp = E; E = E_prev; E_prev = tmp;
   #endif 
 } //end of 'niter' loop at the beginning
 //printf("Ending Program 3 \n");
  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters
  #ifdef _MPI_  
  stats(my_E_prev,mesh_myrows,mesh_mycols,&Linf,&sumSq); 
  double tot_Linf=0.0, tot_SumSq=0.0;
  MPI_Reduce(&Linf, &tot_Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sumSq, &tot_SumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  Linf = tot_Linf;
  L2 = L2Norm(tot_SumSq);

  #else
  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);  
  #endif

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;

  #ifdef _MPI_
  free(my_E);
  free(my_E_prev);
  free(my_R);
  #endif
  //printf("Ending Program\n");
}

void printMat2(const char mesg[], double *E, int m, int n){
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
