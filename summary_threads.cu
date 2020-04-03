#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK_RETURN(value) {\
        cudaError_t _m_cudaStat = value;\
        if (_m_cudaStat != cudaSuccess) {\
        fprintf(stderr, "Error %s at line %d in file %s\n",\
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);\
}}

__global__ void gTest(float* a){
        a[threadIdx.x+blockDim.x*blockIdx.x]=(float)(threadIdx.x+blockDim.x*blockIdx.x);
}

__global__ void sum (float* a, float* b)
{
a[threadIdx.x+blockDim.x*blockIdx.x]+=b[threadIdx.x+blockDim.x*blockIdx.x];
}

int main(int argc, char* argv[]){
        float *da, *db, *ha;
        int num_of_blocks=1<<2, threads_per_block=1<<2;
        int N=num_of_blocks*threads_per_block;
        float elapsedTime;
        
        cudaEvent_t start,stop; 
        cudaEventCreate(&start); 
        cudaEventCreate(&stop); 
        
        ha=(float*)calloc(N, sizeof(float));
        
        CUDA_CHECK_RETURN(cudaMalloc((void**)&da,N*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void**)&db,N*sizeof(float)));
        
        gTest<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(da);
        gTest<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(db);
        
        cudaEventRecord(start,0); 
        sum<<<dim3(num_of_blocks), dim3(threads_per_block)>>>(da,db);	
        cudaEventRecord(stop,0); 
        cudaEventSynchronize(stop);
        CUDA_CHECK_RETURN(cudaGetLastError());
        
        cudaEventElapsedTime(&elapsedTime,start,stop); 
        fprintf(stderr,"gTest took %g\n", elapsedTime);
        cudaEventDestroy(start); 
        cudaEventDestroy(stop); 
        
        CUDA_CHECK_RETURN(cudaMemcpy(ha,da,N*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i=0;i<N;i++)
                printf("%g\n",ha[i]);
        free(ha);
	cudaFree(da);
	cudaFree(db);
        return 0;
}
