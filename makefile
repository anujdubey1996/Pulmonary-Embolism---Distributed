CUDA_PATH = /usr/local/cuda
GDCM_PATH = /usr/include/gdcm-3.0
CC = nvcc
INCLUDES = -I$(CUDA_PATH)/include -I$(GDCM_PATH)  # Adjust the version
LIBS = -L$(GDCM_PATH)/lib -lgdcmDICT -lgdcmMSFF -lcudart
SRC = data_processing.cu
OUT = process_images

all:
	$(CC) $(SRC) -o $(OUT) $(INCLUDES) $(LIBS)

clean:
	rm -f $(OUT)
