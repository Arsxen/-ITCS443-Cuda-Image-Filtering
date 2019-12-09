#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr auto IMG_CHs = 3;
constexpr auto TILE_WIDTH = 32;

void throw_on_cuda_error(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}

__device__ int get_img_byte_position(int img_width, int img_height, int x, int y, int channel) {
	int row = IMG_CHs * img_width * y;
	int w_pos = IMG_CHs * x;
	int position = row + w_pos + channel;
	return position;
}

__global__ void grayscale(unsigned char* img, int img_width, int img_height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < img_width && y < img_height) {
		int sum = 0;
		for (int k = 0; k < channels; k++) {
			sum += img[get_img_byte_position(img_width, img_height, x, y, k)];
		}
		for (int k = 0; k < channels; k++) {
			img[get_img_byte_position(img_width, img_height, x, y, k)] = sum / 3;
		}
	}
}

int main() {
	int width, height, channels;
	unsigned char* data = stbi_load("img.jpg", &width, &height, &channels, IMG_CHs);
	unsigned char* dImg;
	int size = sizeof(char) * width * height * channels;

	try {
		throw_on_cuda_error(cudaMalloc((void**)&dImg, size), __FILE__, __LINE__);
		throw_on_cuda_error(cudaMemcpy(dImg, data, size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
		dim3 dimGrid(TILE_WIDTH, TILE_WIDTH);
		dim3 dimBlock(width / TILE_WIDTH + 1, height / TILE_WIDTH + 1);
		grayscale <<<dimBlock, dimGrid>>> (dImg, width, height, channels);
		throw_on_cuda_error(cudaPeekAtLastError(), __FILE__, __LINE__);
		throw_on_cuda_error(cudaMemcpy(data, dImg, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		throw_on_cuda_error(cudaFree(dImg), __FILE__, __LINE__);
	}
	catch (thrust::system_error & e) {
		std::cerr << e.what() << std::endl;
	}

	for (int i = 0; i < 200; i++) {
		printf("%d ", data[i]);
	}
	printf("\n");

	stbi_write_png("outCuda.png", width, height, channels, data, width * channels);
	stbi_image_free(data);

	return 0;
}