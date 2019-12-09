#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_CHs 3

unsigned char* get_image_byte(unsigned char* img, int img_width, int img_height, int x, int y, int channel);
void grayscale(unsigned char* img, unsigned char* dest, int img_width, int img_height);
void gaussianBlur(unsigned char* img, unsigned char* dest, int img_width, int img_height);
void sobelEdge(unsigned char* img, unsigned char* dest, int img_width, int img_height);

int main() {
    int width, height, channels;
    unsigned char* data = stbi_load("img.jpg", &width, &height, &channels, IMG_CHs);
    unsigned char* out = malloc(sizeof(unsigned char) * width * height * channels);
    grayscale(data, out, width, height);
    stbi_write_png("out.png", width, height, channels, out, width*channels);
    return 0;
}

unsigned char* get_image_byte(unsigned char* img, int img_width, int img_height, int x, int y, int channel) {
    int row = IMG_CHs* img_width * y;
    int w_pos = IMG_CHs * x;
    int position = row + w_pos + channel;
    return &img[position];
}

void grayscale(unsigned char* img, unsigned char* dest, int img_width, int img_height) {
    int i, j, k;
    for (i = 0; i < img_height; i++) {
        for (j = 0; j < img_width; j++) {
            int sum = 0;
            for (k = 0; k < IMG_CHs; k++) {
                sum += *get_image_byte(img, img_width, img_height, j, i, k);
            }
            for (k = 0; k < IMG_CHs; k++) {
                unsigned char* byte = get_image_byte(dest, img_width, img_height, j, i, k);
                *byte = sum/3;
            }
        }
    }
}

void gaussianBlur(unsigned char* img, unsigned char* dest, int img_width, int img_height) {

}

void sobelEdge(unsigned char* img, unsigned char* dest, int img_width, int img_height) {

}