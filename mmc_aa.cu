#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846

__global__ void gpu_mmc(curandState *states, int *global_count, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int local_count = 0;
    
    curandState localState = states[idx];
    
    for(int i = idx; i < n; i += stride) {
        float x = curand_uniform(&localState);
        float y = curand_uniform(&localState);
        if(x*x + y*y <= 1.0f) {
            local_count++;
        }
    }
    
    atomicAdd(global_count, local_count);
    states[idx] = localState;

}

__global__ void setup_kernel(curandState *state, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    curand_init(clock64(), idx, 0, &state[idx]);

}

int main(int argc, char *argv[]) {
    
    int n_points = atoll(argv[1]);
    int n_blocks = 1024;
    int s_blocks = 128;
    int *d_count, h_count = 0;

    float time;

    curandState *d_states;
    cudaError_t nb_error;
    cudaEvent_t start, stop;   

    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;  
    
    int bytes = n_blocks*s_blocks*sizeof(curandState);

    // 
    cudaMalloc(&d_states, bytes);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
    // 
    cudaMalloc(&d_count, sizeof(int));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    // 
    cudaMemset(d_count, 0, sizeof(int));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    // 

    setup_kernel<<<n_blocks, s_blocks>>>(d_states, n_blocks * s_blocks);
    
    gpu_mmc<<<n_blocks, s_blocks>>>(d_states, d_count, n_points);

    cudaDeviceSynchronize();    
        
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    double pi = 4.0 * (double)h_count / (double)n_points;
    double error = fabs(pi - PI)/PI * 100.0;
    
    printf("pi: %.15f\n", pi);
    printf("erro: %.10f%%\n", error);
    printf("tempo: %3.1fms\n", time);
    
    cudaFree(d_states);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    
    cudaFree(d_count);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 5: %s\n", cudaGetErrorString(nb_error));
    
    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));
    
    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(nb_error));
    
    return 0;
    
}