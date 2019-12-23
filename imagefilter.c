//Compile: gcc -o imagefilter imagefilter.c -lm
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/*Struct for Image and Functions*/

typedef struct Image {
    unsigned char* img_data;
    int width;
    int height;
    int channels;
}Image;

Image *newImage(const char *filename);
Image *newEmptyImage(int width, int height, int channels);
void deleteImage(Image *img);
int _Image_get_position(Image *img, int x, int y, int channel);
unsigned char getImageValue(Image *img, int x, int y, int channel);
void setImageValue(Image *img, int val, int x, int y, int channel);
void timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

/*Image Filtering*/
void gaussianBlur(Image *src, Image *dest){
    double GKernel[5][5]; // Kernel size 5*5
    double stdi = 1.0; // starndard devision = 1
    double r, s = 2.0 * stdi * stdi;
    double sum = 0.0;
    int x,y,i,j,k,kRow,kCol;
    for (x = -2; x <= 2; x++) {
        for (y = -2; y <= 2; y++) {
            r = x * x + y * y;
            GKernel[x + 2][y + 2] = (exp(-r / s)) / (M_PI * s);
            sum += GKernel[x + 2][y + 2];
        }
    }
    for (i = 0; i < 5; ++i){
        for (j = 0; j < 5; ++j){
            GKernel[i][j] /= sum;
        }
    }
    // Filter image with Gaussian Blur
    int rows=src->width;
    int cols=src->height;


    int verticleImageBound=(5-1)/2;
    int horizontalImageBound=(5-1)/2;
    for (i = 0+verticleImageBound; i < rows-verticleImageBound; i++) {
        for (j = 0+horizontalImageBound; j < cols-horizontalImageBound; j++) {
            for (k = 0; k < src->channels; k++) {
                float value=0.0;
                for(kRow=0;kRow<5;kRow++){
                    for(kCol=0;kCol<5;kCol++){
                        //multiply pixel value with corresponding gaussian kernal value
                        float pixel = getImageValue(src, kRow+i-verticleImageBound, kCol+j-horizontalImageBound, k)*GKernel[kRow][kCol];
                        value+=pixel;
                    }
                }
                int value_floor =  floor(value);
                setImageValue(dest, value_floor, i, j, k);
            }
        }
    }
}
void sobelEdge(Image *src, Image *dest) {
    int x_kernel[3][3] = { {-1,0,1},
                           {-2,0,2},
                           {-1,0,1} };
    int y_kernel[3][3] = { {1,2,1},
                           {0,0,0},
                           {-1,-2,-1} };
    int x,y,k;
    for (x = 0; x < src->width; x++) {
        for (y = 0; y < src->width; y++) {
            for (k = 0; k < dest->channels; k++) {
                int x2, y2;
                int horizontal_val = 0;
                int vertical_val = 0;
                for (x2 = -1; x2 <= 1; x2++) {
                    for (y2 = -1; y2 <= 1; y2++) {
                        int new_posx = x + x2;
                        int new_posy = y + y2;
                        //Horizontal Sobel Edge Dectecion
                        horizontal_val += x_kernel[x2 + 1][y2 + 1] * getImageValue(src, new_posx, new_posy, k);
                        //Vertical Sobel Edge Dectection
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
}

/************************Main*******************************************/

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Please specify filename\n");
        printf("Usage: ./imagefilter <filename>\n");
        printf("<filename>: File name of an image (JPEG only)\n");
        return -1;
    }

    //Time measurement
    struct timeval tvBegin, tvEnd, tvDiff;

    Image *src = newImage(argv[1]);
    if (src == NULL) {
        printf("Cannot open file! (This program accept only JPEG)\n");
        return -1;
    }
    Image *destSobel = newEmptyImage(src->width, src->height, 3);
    Image *destGaussian = newEmptyImage(src->width, src->height, 3);

    // begin
    gettimeofday(&tvBegin, NULL);

    sobelEdge(src,destSobel);
    gaussianBlur(src,destGaussian);

    //end
    gettimeofday(&tvEnd, NULL);

    // diff
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    printf("Computation Time: %ld.%06ld seconds\n", tvDiff.tv_sec, tvDiff.tv_usec);

    //Write result to file
    stbi_write_jpg("Out_Sobel.jpg", destSobel->width, destSobel->height, destSobel->channels, destSobel->img_data, 100);
    stbi_write_jpg("Out_GaussianBlur.jpg", destGaussian->width, destGaussian->height, destGaussian->channels, destGaussian->img_data, 100);

    //Free Memory
    deleteImage(src);
    deleteImage(destSobel);
    deleteImage(destGaussian);
    return 0;
}

/************************EndMain*******************************************/

Image *newImage(const char *filename) {

    //Allocate Struct Image
    Image *retImg = malloc(sizeof(Image));
    if (retImg == NULL)
        return NULL;

    //Load the image data
    int load_width, load_height, load_channels;
    unsigned char *data = stbi_load(filename, &load_width, &load_height, &load_channels, 0);
    if (data == NULL) {
        free(retImg);
        return NULL;
    }

    retImg->img_data = data;
    retImg->width = load_width;
    retImg->height = load_height;
    retImg->channels = load_channels;

    return retImg;
}

Image *newEmptyImage(int width, int height, int channels) {
    Image *retImg = malloc(sizeof(Image));
    if (retImg == NULL)
        return NULL;

    int size = width * height * channels;
    retImg->img_data = malloc(size * sizeof(unsigned char));
    if (retImg->img_data == NULL) {
        free(retImg);
        return NULL;
    }

    retImg->width = width;
    retImg->height = height;
    retImg->channels = channels;

    return retImg;
}

void deleteImage(Image *img) {
    if (img != NULL) {
        stbi_image_free(img->img_data);
        free(img);
    }
}

int _Image_get_position(Image *img, int x, int y, int channel) {
    int row = img->channels * img->width * y;
    int w_pos = img->channels * x;
    int position = row + w_pos + channel;
    return position;
}

unsigned char getImageValue(Image *img, int x, int y, int channel) {
    //Handle out of bound like image is got padding with 0
    if (x < 0 || y < 0 || channel < 0 || x >= img->width || y >= img->height || channel >= img->channels)
        return 0;

    int pos = _Image_get_position(img, x, y, channel);
    return img->img_data[pos];
}

void setImageValue(Image *img, int val, int x, int y, int channel) {
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

void timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;
}