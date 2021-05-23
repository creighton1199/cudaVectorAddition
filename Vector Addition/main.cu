/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <time.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;

    float* Ad, *Bd, *Cd; 


    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    
    

    float* A_h = (float*) malloc(n*sizeof(float));
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc(n*sizeof(float));
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc(n*sizeof(float));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    if (cudaSuccess!=cudaMalloc((void**)&Ad, n*sizeof(float)))
    {
        printf("Error in memory allocation/n");
        exit(-1);
    }
    if (cudaSuccess!=cudaMalloc((void**)&Bd, n*sizeof(float)))
    {
        printf("Error allocating memory\n");
        exit(-1);
    }
    if (cudaSuccess!=cudaMalloc((void**)&Cd, n*sizeof(float)))
    {
        printf("Error allocating memory\n");
        exit(-1);
    }










    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    int size=n*sizeof(float);

    if (cudaSuccess !=  cudaMemcpy(Ad,A_h, size, cudaMemcpyHostToDevice))
    {
        printf("Error copying memory to data\n");
        exit(-1);
    }
    if (cudaSuccess != cudaMemcpy(Bd,B_h, size, cudaMemcpyHostToDevice))
    {
        printf("Error copying memory to data\n");
        exit(-1);
    }




    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    dim3 dimBlock(16,16,1);
    dim3 dimGrid(ceil(n/16),1,1);

    vecAddKernel<<<dimGrid,dimBlock>>>(Ad,Bd, Cd, n);



    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    
    if (cudaSuccess != cudaMemcpy(C_h,Cd, size, cudaMemcpyDeviceToHost))
    {
        printf("Error copying data to host");
        exit(-1);
    }
    


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

    if (cudaSuccess != cudaFree(Ad))
    {
        printf("Error releasing cuda");
        exit(-1);
    }
    if (cudaSuccess != cudaFree(Bd))
    {
        printf("Error releasing cuda");
        exit(-1);
    }
    if (cudaSuccess != cudaFree(Cd))
    {
        printf("Error releasing cuda");
        exit(-1);
    }


    return 0;

}

