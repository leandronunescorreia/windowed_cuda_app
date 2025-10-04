#include "cuda_gfx.h"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>


#define MEASURE_DURATION_AND_RETURN(start) \
	{ \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::nano> duration = end - start; \
    return duration.count(); \
	}

__global__ void process_image(uint8_t* img, int32_t width, int32_t height) {
	uint8_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t y = blockIdx.y * blockDim.y + threadIdx.y;

	int idx = (x + y * width) * 3;

	img[idx] =100;
	img[idx + 1] = 150;
	img[idx + 2] = 255;

}

double toDevice(cuda_gfx_t* context, uint8_t* rgb, int32_t width, int32_t height) {
	auto start = std::chrono::high_resolution_clock::now();

	size_t buffer_size = (width * height) * sizeof(uint8_t) * 3;

	CUDA_CHECK_RETURN(cudaMalloc(&context->buffer, buffer_size));
	CUDA_CHECK_RETURN(cudaMemcpy(context->buffer, rgb, buffer_size, cudaMemcpyHostToDevice));

	context->grid_size = dim3((width + 15) / 16, (height + 15) / 16);
	context->block_size = dim3(16, 16);
	context->width = width;
	context->height = height;

	MEASURE_DURATION_AND_RETURN(start);
}

double toHost(cuda_gfx_t* context, uint8_t* rgb) {
	auto start = std::chrono::high_resolution_clock::now();
	size_t buffer_size = (context->width * context->height) * sizeof(uint8_t) * 3;

	CUDA_CHECK_RETURN(cudaMemcpy(rgb, context->buffer, buffer_size, cudaMemcpyDeviceToHost));

	MEASURE_DURATION_AND_RETURN(start);
}


double run(cuda_gfx_t* context) {
	auto start = std::chrono::high_resolution_clock::now();

	process_image<<<context->grid_size, context->block_size>>>(context->buffer, context->width, context->height);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	MEASURE_DURATION_AND_RETURN(start);
}

void cleanup(cuda_gfx_t* context) {
	if (context && context->buffer) {
		cudaFree(context->buffer);
		context->buffer = nullptr;
	}
}
