#include <cuda_runtime.h>
#include "image_processing.h"

__global__ void processImageKernel(float* input, float* output, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width * height) {
        output[idx] = 255 - input[idx];  // Example operation
    }
}

extern "C" void processImageCUDA(float* input, float* output, int width, int height) {
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    dim3 threads(16, 16);
    processImageKernel<<<blocks, threads>>>(d_input, d_output, width, height);
    
    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}
