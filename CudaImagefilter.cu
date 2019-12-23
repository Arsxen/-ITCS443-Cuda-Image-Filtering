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

#define TILE_WIDTH 32

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

/*Struct for Image and Helper Functions*/

typedef struct Image {
	unsigned char* img_data;
	int width;
	int height;
	int channels;
}Image;

__host__ Image* newImage(const char *filename);
__host__ Image* newEmptyImage(int width, int height, int channels);
__host__ void deleteImage(Image *img);
__host__ __device__ int _Image_get_position(Image *img, int x, int y, int channel);
__host__ __device__ unsigned char getImageValue(Image *img, int x, int y, int channel);
__host__ __device__ void setImageValue(Image *img, int val, int x, int y, int channel);

/*Image Filtering*/
__device__ unsigned char pixelAdd(int a, int b) {
	int res = a + b;
	if (res > 255)
		return (unsigned char)255;
	else
		return (unsigned char)res;
}

__global__ void grayscale(Image *src, Image *dest) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int sum = 0;
	int k;
	if (x < src->width && y < src->height) {
		for (k = 0; k < src->channels; k++) {
			sum += getImageValue(src, x, y, k);
		}
		int new_val = sum / dest->channels;
		for (k = 0; k < dest->channels; k++) {
			setImageValue(dest, new_val, x, y, k);
		}
	}
}

__global__ void gaussianBlur(Image *src, Image *dest){
	// generate GaussianBlur Kernel
	double GKernel[5][5] = {
	{0.002969, 0.013306, 0.021938, 0.013306, 0.002969},
	{0.013306, 0.059634, 0.098320, 0.059634, 0.013306},
	{0.021938, 0.098320, 0.162103, 0.098320, 0.021938},
	{0.013306, 0.059634, 0.098320, 0.059634, 0.013306},
	{0.002969, 0.013306, 0.021938, 0.013306, 0.002969}
	}; // Kernel size 5*5
	int rows = src->width;
	int cols = src->height;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int k;
	// Filter image with Gaussian Blur
	if (x < rows - 2 && y < cols - 2 && x>1 && y>1) {
		for (k = 0; k < src->channels; k++) {
			float value = 0.0;
			for (int kRow = 0; kRow < 5; kRow++) {
				for (int kCol = 0; kCol < 5; kCol++) {
					//multiply pixel value with corresponding gaussian kernal value
					float pixel = getImageValue(src, kRow + x - 2, kCol + y - 2, k) * GKernel[kRow][kCol];
					value += pixel;
				}
			}
			int value_floor = floor(value);
			setImageValue(dest, value_floor, x, y, k);
		}
	}
}

__global__ void sobelEdge(Image *src, Image *dest){
	int x_kernel[3][3] = { {-1,0,1},
							{-2,0,2},
							{-1,0,1} };
	int y_kernel[3][3] = { {1,2,1},
							{0,0,0},
							{-1,-2,-1} };

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < src->width && y < src->height) {
		int k;
		for (k = 0; k < dest->channels; k++) {
			int x2, y2;
			int horizontal_val = 0;
			int vertical_val = 0;
			for (x2 = -1; x2 <= 1; x2++) {
				for (y2 = -1; y2 <= 1; y2++) {
					int new_posx = x + x2;
					int new_posy = y + y2;
					horizontal_val += x_kernel[x2 + 1][y2 + 1] * getImageValue(src, new_posx, new_posy, k);
					vertical_val += y_kernel[x2 + 1][y2 + 1] * getImageValue(src, new_posx, new_posy, k);
				}
			}

			//compute magnitude
			double pow_horizontal = pow((double)horizontal_val, 2.0);
			double pow_vertical = pow((double)vertical_val, 2.0);
			double magnitude = sqrt(pow_horizontal + pow_vertical);
			int magnitude_floor = floor(magnitude);
			setImageValue(dest, magnitude_floor, x, y, k);
		}
	}
	
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		printf("Please specify filename\n");
		printf("Usage: ./CudaImagefilter <filename>\n");
		printf("<filename>: File name of an image (JPEG only)\n");
		return -1;
	}
	Image *src = newImage("img.jpg");
	Image *destSobel = newEmptyImage(src->width, src->height, src->channels);
	Image *destGaussian = newEmptyImage(src->width, src->height, src->channels);

	//Time Measurement
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Begin
	cudaEventRecord(start, 0);

	//Execute Kernel
	dim3 dimGird(src->width/TILE_WIDTH + 1, src->height/TILE_WIDTH + 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	gaussianBlur<<<dimGird, dimBlock>>>(src, destGaussian);
	sobelEdge<<<dimGird, dimBlock>>>(src, destSobel);
	cudaDeviceSynchronize();

	//End
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//Diff
	printf("Computation Time:  %f seconds \n", time/1000.0f);

	stbi_write_jpg("Out_CudaSobel.jpg", destSobel->width, destSobel->height, destSobel->channels, destSobel->img_data, 100);
	stbi_write_jpg("Out_CudaGaussianBlur.jpg", destGaussian->width, destGaussian->height, destGaussian->channels, destGaussian->img_data, 100);

	deleteImage(src);
	deleteImage(destSobel);
	deleteImage(destGaussian);

	return 0;
}

/*****************************Image Helper Function***********************************************/
__host__ Image* newImage(const char *filename) {
	int code;
	//Allocate Struct Image
	Image* retImg;
	code = cudaMallocManaged(&retImg, sizeof(retImg));
	if (code != cudaSuccess)
		return NULL;

	//Load the image data
	int load_width, load_height, load_channels;
	unsigned char* data = stbi_load(filename, &load_width, &load_height, &load_channels, 0);
	if (data == NULL) {
		cudaFree(retImg);
		return NULL;
	}

	//Allocate image data
	int size = sizeof(unsigned char) * load_height * load_width * load_channels;
	code = cudaMallocManaged(&retImg->img_data, size);
	if (code != cudaSuccess) {
		cudaFree(retImg);
		stbi_image_free(data);
		return NULL;
	}

	//Copy Image data to unified memory
	memcpy(retImg->img_data, data, size);

	retImg->width = load_width;
	retImg->height = load_height;
	retImg->channels = load_channels;

	stbi_image_free(data);

	return retImg;
}

__host__ Image* newEmptyImage(int width, int height, int channels) {
	int code;

	//Allocate Struct Image
	Image* retImg;
	code = cudaMallocManaged(&retImg, sizeof(retImg));
	if (code != cudaSuccess)
		return NULL;

	int size = width * height * channels;
	code = cudaMallocManaged(&retImg->img_data, size);
	if (code != cudaSuccess) {
		cudaFree(retImg);
		return NULL;
	}

	retImg->width = width;
	retImg->height = height;
	retImg->channels = channels;

	return retImg;
}

__host__ void deleteImage(Image *img) {
	if (img != NULL) {
		cudaDeviceSynchronize();
		cudaFree(img->img_data);
		cudaFree(img);
	}
}

__host__ __device__ int _Image_get_position(Image *img, int x, int y, int channel) {
	int row = img->channels * img->width * y;
	int w_pos = img->channels * x;
	int position = row + w_pos + channel;
	return position;
}

__host__ __device__ unsigned char getImageValue(Image *img, int x, int y, int channel) {
	//Handle out of bound like image is got padding with 0
	if (x < 0 || y < 0 || channel < 0 || x >= img->width || y >= img->height || channel >= img->channels)
		return 0;

	int pos = _Image_get_position(img, x, y, channel);
	return img->img_data[pos];
}

__host__ __device__ void setImageValue(Image *img, int val, int x, int y, int channel) {
	//Check Out of Bound
	if (x < 0 || y < 0 || channel < 0 || x >= img->width || y >= img->height || channel >= img->channels)
		return;

	int pos = _Image_get_position(img, x, y, channel);
	if (val > 255)
		img->img_data[pos] = 255;
	else if (val < 0)
		img->img_data[pos] = 0;
	else
		img->img_data[pos] = val;
}

/*****************************End Image Helper Function***********************************************/
