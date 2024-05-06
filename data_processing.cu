#include <iostream>
#include <cuda_runtime.h>

#include <gdcmImageReader.h>
#include <gdcmImage.h>
#include <gdcmDataElement.h>

using namespace std;

void loadDICOMImage(const char* filename, float *imageBuffer, int imageSize) {
    gdcm::ImageReader reader;
    reader.SetFileName(filename);
    if (!reader.Read()) {
        std::cerr << "Failed to read DICOM image: " << filename << std::endl;
        exit(1);
    }

    const gdcm::Image &image = reader.GetImage();
    if (image.GetPhotometricInterpretation() != gdcm::PhotometricInterpretation::MONOCHROME2) {
        std::cerr << "Unsupported Photometric Interpretation." << std::endl;
        exit(1);
    }

    image.GetBuffer((char*)imageBuffer);
}


// Windowing Kernel
__global__ void windowingKernel(float *images, float *output, int numImages, int width, int height, float lowerBound, float upperBound) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages * width * height) {
        float pixel = images[idx];
        if (pixel < lowerBound) {
            output[idx] = 0;
        } else if (pixel > upperBound) {
            output[idx] = 255;
        } else {
            output[idx] = 255 * (pixel - lowerBound) / (upperBound - lowerBound);
        }
    }
}

// Median Filter Kernel
__global__ void medianFilter(float *images, float *output, int numImages, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int imageIdx = idx / (width * height);
    int pixelIdx = idx % (width * height);
    int x = pixelIdx % width;
    int y = pixelIdx / width;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && idx < numImages * width * height) {
        float neighbors[9];
        int nIdx = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                neighbors[nIdx++] = images[imageIdx * width * height + (y + dy) * width + (x + dx)];
            }
        }

        // Insertion sort for simplicity
        for (int i = 1; i < 9; i++) {
            float key = neighbors[i];
            int j = i - 1;
            while (j >= 0 && neighbors[j] > key) {
                neighbors[j + 1] = neighbors[j];
                j = j - 1;
            }
            neighbors[j + 1] = key;
        }

        output[idx] = neighbors[4]; // Median value
    }
}

__global__ void sobelEdgeDetection(float *images, float *output, int numImages, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numImages * width * height) {
        int imageIdx = idx / (width * height);
        int pixelIdx = idx % (width * height);
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            float Gx = images[imageIdx * width * height + (y - 1) * width + (x - 1)] * -1.0f +
                       images[imageIdx * width * height + (y - 1) * width + (x + 1)] * 1.0f +
                       images[imageIdx * width * height + y * width + (x - 1)] * -2.0f +
                       images[imageIdx * width * height + y * width + (x + 1)] * 2.0f +
                       images[imageIdx * width * height + (y + 1) * width + (x - 1)] * -1.0f +
                       images[imageIdx * width * height + (y + 1) * width + (x + 1)] * 1.0f;
            float Gy = images[imageIdx * width * height + (y - 1) * width + (x - 1)] * -1.0f +
                       images[imageIdx * width * height + (y - 1) * width + x] * -2.0f +
                       images[imageIdx * width * height + (y - 1) * width + (x + 1)] * -1.0f +
                       images[imageIdx * width * height + (y + 1) * width + (x - 1)] * 1.0f +
                       images[imageIdx * width * height + (y + 1) * width + x] * 2.0f +
                       images[imageIdx * width * height + (y + 1) * width + (x + 1)] * 1.0f;

            float edgeStrength = sqrtf(Gx * Gx + Gy * Gy);
            output[idx] = edgeStrength;
        } else {
            // Handle boundaries: set edge strength to zero
            output[idx] = 0.0f;
        }
    }
}


// Function to check CUDA errors
void checkCudaError(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// Function to allocate and initialize memory
void allocateMemory(float **inputImages, float **processedImages, float **outputImages, int totalSize) {
    // Allocate device memory for input, processed, and output images
    checkCudaError(cudaMalloc(inputImages, totalSize * sizeof(float)), "Failed to allocate device memory for input images");
    checkCudaError(cudaMalloc(processedImages, totalSize * sizeof(float)), "Failed to allocate device memory for processed images");
    checkCudaError(cudaMalloc(outputImages, totalSize * sizeof(float)), "Failed to allocate device memory for output images");

    // Example: Initialize input images with random values (for demonstration)
    float *h_inputImages = new float[totalSize];
    for (int i = 0; i < totalSize; i++) {
        h_inputImages[i] = static_cast<float>(rand() % 256);
    }

    // Copy data from host to device
    checkCudaError(cudaMemcpy(*inputImages, h_inputImages, totalSize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data from host to device");

    // Cleanup host memory
    delete[] h_inputImages;
}

#include <gdcmImageReader.h>
#include <gdcmImage.h>

bool loadDICOMImage(const std::string& filename, float*& imageBuffer, int& width, int& height) {
    gdcm::ImageReader reader;
    reader.SetFileName(filename.c_str());
    if (!reader.Read()) {
        std::cerr << "Could not read DICOM image: " << filename << std::endl;
        return false;
    }

    const gdcm::Image &image = reader.GetImage();
    width = image.GetColumns();
    height = image.GetRows();

    if (imageBuffer != nullptr) {
        delete[] imageBuffer;
    }
    imageBuffer = new float[width * height];

    image.GetBuffer((char*)imageBuffer);

    return true;
}

void processImages(const std::vector<std::string>& imagePaths, int numImagesToProcess) {
    float *d_inputImage, *d_processedImage, *d_outputImage;

    for (int i = 0; i < numImagesToProcess; ++i) {
        int width, height;
        float *h_inputImage = nullptr;

        if (!loadDICOMImage(imagePaths[i], h_inputImage, width, height)) {
            continue;  // Skip this image if there's a problem loading it
        }

        int imageSize = width * height;

        // Allocate device memory
        checkCudaError(cudaMalloc(&d_inputImage, imageSize * sizeof(float)), "Allocate input image");
        checkCudaError(cudaMalloc(&d_processedImage, imageSize * sizeof(float)), "Allocate processed image");
        checkCudaError(cudaMalloc(&d_outputImage, imageSize * sizeof(float)), "Allocate output image");

        // Copy data from host to device
        checkCudaError(cudaMemcpy(d_inputImage, h_inputImage, imageSize * sizeof(float), cudaMemcpyHostToDevice), "Copy input image to device");

        // Configure kernel dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (imageSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch processing kernels (modify as needed)
        windowingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inputImage, d_processedImage, width, height, 50.0f, 200.0f);
        medianFilter<<<blocksPerGrid, threadsPerBlock>>>(d_processedImage, d_inputImage, width, height);
        sobelEdgeDetection<<<blocksPerGrid, threadsPerBlock>>>(d_inputImage, d_outputImage, width, height);

        // Copy back the results
        float *h_outputImage = new float[imageSize];
        checkCudaError(cudaMemcpy(h_outputImage, d_outputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost), "Copy output image to host");

        // Process results as necessary

        // Clean up
        cudaFree(d_inputImage);
        cudaFree(d_processedImage);
        cudaFree(d_outputImage);
        delete[] h_inputImage;
        delete[] h_outputImage;
    }
}

int main() {
    vector<string> dicomFilePaths;
    collectDicomFiles("/mnt/data/filtered_data/train/train", dicomFilePaths);
    int numImagesToProcess = min(10, static_cast<int>(dicomFilePaths.size()));  // Process up to 10 images

    processImages(dicomFilePaths, numImagesToProcess);

    return 0;
}
