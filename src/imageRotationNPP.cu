// src/imageRotationNPP.cu
// A standalone CUDA + stb_image grayscale converter.
// Accepts exactly 13 argv entries: 
//   prog --input in.png --output out.png --block Bx By Bz --grid Gx Gy Gz

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel: average RGB→gray
__global__ void grayKernel(
    const unsigned char* __restrict__ in,
    unsigned char*       __restrict__ out,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * channels;
    unsigned int sum = in[idx + 0];
    if (channels > 1) sum += in[idx + 1];
    if (channels > 2) sum += in[idx + 2];
    out[y * width + x] = sum / channels;
}

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --input  <in.png>  --output <out.png>  \\\n"
        "           --block  Bx By Bz  --grid   Gx Gy Gz\n",
        prog
    );
}

int main(int argc, char** argv) {
    // Expect exactly 13 args
    if (argc != 13 ||
        std::strcmp(argv[1], "--input")  ||
        std::strcmp(argv[3], "--output") ||
        std::strcmp(argv[5], "--block")  ||
        std::strcmp(argv[9], "--grid"))
    {
        printUsage(argv[0]);
        return 1;
    }

    const char* inFile  = argv[2];
    const char* outFile = argv[4];

    dim3 block(
        std::atoi(argv[6]),
        std::atoi(argv[7]),
        std::atoi(argv[8])
    );
    dim3 grid(
        std::atoi(argv[10]),
        std::atoi(argv[11]),
        std::atoi(argv[12])
    );

    int w, h, channels;
    unsigned char* h_img = stbi_load(inFile, &w, &h, &channels, 0);
    if (!h_img) {
        std::fprintf(stderr, "ERROR: failed to load '%s'\n", inFile);
        return 1;
    }

    size_t inBytes  = size_t(w) * h * channels;
    size_t outBytes = size_t(w) * h;

    std::printf("Copy input data from the host memory to the CUDA device\n");
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  inBytes);
    cudaMalloc(&d_out, outBytes);
    cudaMemcpy(d_in, h_img, inBytes, cudaMemcpyHostToDevice);

    std::printf("CUDA kernel launch with %d blocks of %d threads\n",
                int(grid.x*grid.y*grid.z),
                int(block.x*block.y*block.z));
    grayKernel<<<grid, block>>>(d_in, d_out, w, h, channels);
    cudaDeviceSynchronize();

    std::printf("Copy output data from the CUDA device to the host memory\n");
    unsigned char* h_out = (unsigned char*)std::malloc(outBytes);
    cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost);

    // Write single-channel PNG
    stbi_write_png(outFile, w, h, 1, h_out, w);

    // Cleanup
    stbi_image_free(h_img);
    std::free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
