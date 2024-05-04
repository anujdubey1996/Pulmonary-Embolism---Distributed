#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using namespace std;

__global__ void normalizeKernel(float *input, float *output, int numPixels, float minVal, float maxVal) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numPixels) {
        output[idx] = (input[idx] - minVal) / (maxVal - minVal);
    }
}

__device__ float bilinearInterpolate(float tl, float tr, float bl, float br, float x_frac, float y_frac) {
    float top = tl + x_frac * (tr - tl);
    float bottom = bl + x_frac * (br - bl);
    return top + y_frac * (bottom - top);
}

__global__ void resizeKernel(float *input, float *output, int oldWidth, int oldHeight, int newWidth, int newHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < newWidth && y < newHeight) {
        float gx = x * (oldWidth - 1) / (float)(newWidth - 1);
        float gy = y * (oldHeight - 1) / (float)(newHeight - 1);
        
        int gxi = (int)gx;
        int gyi = (int)gy;

        float x_frac = gx - gxi;
        float y_frac = gy - gyi;

        int idx00 = gyi * oldWidth + gxi;
        int idx01 = gyi * oldWidth + gxi + 1;
        int idx10 = (gyi + 1) * oldWidth + gxi;
        int idx11 = (gyi + 1) * oldWidth + gxi + 1;

        float result = bilinearInterpolate(input[idx00], input[idx01], input[idx10], input[idx11], x_frac, y_frac);
        output[y * newWidth + x] = result;
    }
}

__global__ void sobelFilterKernel(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float x_weight = 0;
        float y_weight = 0;

        int offset = y * width + x;
        x_weight = input[offset - width - 1] * -1.0 + input[offset - width + 1] * 1.0 +
                   input[offset - 1] * -2.0 + input[offset + 1] * 2.0 +
                   input[offset + width - 1] * -1.0 + input[offset + width + 1] * 1.0;
        y_weight = input[offset - width - 1] * -1.0 + input[offset - width] * -2.0 + input[offset - width + 1] * -1.0 +
                   input[offset + width - 1] * 1.0 + input[offset + width] * 2.0 + input[offset + width + 1] * 1.0;

        output[offset] = sqrt(x_weight * x_weight + y_weight * y_weight);
    }
}



int main() {
    // Image dimensions
    const int oldWidth = 512, oldHeight = 512;
    const int newWidth = 256, newHeight = 256;
    const int numPixels = oldWidth * oldHeight;
    const int newNumPixels = newWidth * newHeight;

    // Allocate host memory
    float *h_input = new float[numPixels];
    float *h_normalized = new float[numPixels];
    float *h_resized = new float[newNumPixels];
    float *h_edges = new float[newNumPixels];

    // Initialize input data
    for (int i = 0; i < numPixels; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 255.0f; // Random float values
    }

    // Allocate device memory
    float *d_input, *d_normalized, *d_resized, *d_edges;
    cudaMalloc((void**)&d_input, numPixels * sizeof(float));
    cudaMalloc((void**)&d_normalized, numPixels * sizeof(float));
    cudaMalloc((void**)&d_resized, newNumPixels * sizeof(float));
    cudaMalloc((void**)&d_edges, newNumPixels * sizeof(float));
    cout<<"Allocated Memory"<<endl;
    // Copy data to device
    cudaMemcpy(d_input, h_input, numPixels * sizeof(float), cudaMemcpyHostToDevice);

    // Launch normalization kernel
    int blockSize = 256;
    int numBlocks = (numPixels + blockSize - 1) / blockSize;
    normalizeKernel<<<numBlocks, blockSize>>>(d_input, d_normalized, numPixels, 0.0f, 255.0f);

    // Launch resize kernel
    dim3 block(16, 16);
    dim3 grid((newWidth + 15) / 16, (newHeight + 15) / 16);
    resizeKernel<<<grid, block>>>(d_normalized, d_resized, oldWidth, oldHeight, newWidth, newHeight);

    // Launch Sobel edge detection kernel
    sobelFilterKernel<<<grid, block>>>(d_resized, d_edges, newWidth, newHeight);

    // Copy results back to host
    cudaMemcpy(h_normalized, d_normalized, numPixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resized, d_resized, newNumPixels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edges, d_edges, newNumPixels * sizeof(float), cudaMemcpyDeviceToHost);

    // Output some sample data
    std::cout << "Sample outputs:" << std::endl;
    std::cout << "Normalized: " << h_normalized[0] << ", " << h_normalized[1] << std::endl;
    std::cout << "Resized: " << h_resized[0] << ", " << h_resized[1] << std::endl;
    std::cout << "Edges: " << h_edges[0] << ", " << h_edges[1] << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_normalized);
    cudaFree(d_resized);
    cudaFree(d_edges);

    // Free host memory
    delete[] h_input;
    delete[] h_normalized;
    delete[] h_resized;
    delete[] h_edges;

    return 0;
}
