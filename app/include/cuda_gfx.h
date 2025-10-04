#ifndef CUDA_GFX
#define CUDA_GFX

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio> // for stderr


#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	} }


typedef struct {
	int32_t height;
	int32_t width;
	dim3 grid_size;
	dim3 block_size;
	uint8_t* buffer;
} cuda_gfx_t;


__global__ void process_image(uint8_t* img, int32_t width, int32_t height);

double toDevice(cuda_gfx_t* context, uint8_t* rgb, int32_t width, int32_t height);
double toHost(cuda_gfx_t* context, uint8_t* rgb);
double run(cuda_gfx_t* context);

void cleanup(cuda_gfx_t* context);

#endif