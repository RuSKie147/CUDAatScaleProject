// src/main.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

// Simple CUDA grayscale: in = uchar3 RGB, out = single uchar gray
__global__ void grayscaleKernel(
    const unsigned char* __restrict__ in,
    unsigned char*       __restrict__ out,
    int width, int height, int channels)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y*width + x) * channels;
    unsigned char r = in[idx+0];
    unsigned char g = (channels>1 ? in[idx+1] : r);
    unsigned char b = (channels>2 ? in[idx+2] : r);
    out[y*width + x] = (r + g + b) / 3;
}

// Parse: --input, --output, --block Bx By Bz, --grid Gx Gy Gz
void parseArgs(int argc, char** argv,
               const char*& inPath,
               const char*& outPath,
               dim3& block, dim3& grid)
{
    inPath = nullptr; outPath = nullptr;
    block = dim3(16,16,1);
    grid  = dim3(32,32,1);

    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"--input"))  inPath = argv[++i];
        if (!strcmp(argv[i],"--output")) outPath = argv[++i];
        if (!strcmp(argv[i],"--block")) {
            block.x = atoi(argv[++i]);
            block.y = atoi(argv[++i]);
            block.z = atoi(argv[++i]);
        }
        if (!strcmp(argv[i],"--grid")) {
            grid.x = atoi(argv[++i]);
            grid.y = atoi(argv[++i]);
            grid.z = atoi(argv[++i]);
        }
    }
    if (!inPath || !outPath) {
        fprintf(stderr,"Usage: %s --input in.png --output out.png "
                       "--block Bx By Bz --grid Gx Gy Gz\n", argv[0]);
        exit(1);
    }
}

int main(int argc, char** argv)
{
    const char *inFile, *outFile;
    dim3 block, grid;
    parseArgs(argc, argv, inFile, outFile, block, grid);

    int w,h,channels;
    unsigned char *h_img = stbi_load(inFile, &w,&h,&channels,0);
    if (!h_img) {
        fprintf(stderr,"Failed to load %s\n", inFile);
        return 1;
    }

    size_t inBytes  = size_t(w)*h*channels;
    size_t outBytes = size_t(w)*h;

    printf("Copy input data from the host memory to the CUDA device\n");
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  inBytes);
    cudaMalloc(&d_out, outBytes);
    cudaMemcpy(d_in, h_img, inBytes, cudaMemcpyHostToDevice);

    printf("CUDA kernel launch with %d blocks of %d threads\n",
           block.x*grid.x*grid.y*grid.z,
           block.x*block.y*block.z);

    grayscaleKernel<<<grid, block>>>(d_in, d_out, w, h, channels);
    cudaDeviceSynchronize();

    printf("Copy output data from the CUDA device to the host memory\n");
    unsigned char *h_out = (unsigned char*)malloc(outBytes);
    cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost);

    // write single-channel gray PNG
    stbi_write_png(outFile, w, h, 1, h_out, w);

    // cleanup
    stbi_image_free(h_img);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
