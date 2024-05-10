# cython: language_level=3

import numpy as np
cimport numpy as cnp

# Declare the C function from the header
cdef extern from "image_processing.h":
    void processImage(float* input, float* output, int width, int height)

def process_image(cnp.ndarray[cnp.float32_t, ndim=2] input_image):
    # Ensure the input NumPy array is C-contiguous in memory for compatibility.
    if not input_image.flags['C_CONTIGUOUS']:
        input_image = np.ascontiguousarray(input_image, dtype=np.float32)
    
    cdef int width = input_image.shape[1]
    cdef int height = input_image.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] output_image = np.empty_like(input_image)

    # Call the C function with pointers to the data of the NumPy arrays.
    processImage(<float*> input_image.data, <float*> output_image.data, width, height)
    
    return output_image
