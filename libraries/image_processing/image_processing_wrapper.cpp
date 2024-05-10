#include <iostream>
#include <vector>
#include "image_processing.h"

extern "C" void processImage(float* input, float* output, int width, int height) {
    void processImageCUDA(float* input, float* output, int width, int height);
    processImageCUDA(input, output, width, height);
}