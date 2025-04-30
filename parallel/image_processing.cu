#include "image_processing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Gestione errori CUDA
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(error) << std::endl; \
        cudaDeviceReset(); \
        return false; \
    } \
} while(0)

// Definizione costante per dimensione blocco CUDA
#define BLOCK_SIZE 16

// Versione parallela per elaborare tutti e tre i canali contemporaneamente
// Memoria costante per il kernel
__constant__ float d_kernelConst[225]; // supporta kernel fino a 15x15

// Kernel CUDA: applica convoluzione usando texture memory e kernel in memoria costante
__global__ void filterWithTexture(cudaTextureObject_t texImage, float* __restrict__ output, int width, int height, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = kSize / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        for (int j = -r; j <= r; ++j) {
            for (int i = -r; i <= r; ++i) {
                float val = tex2D<float>(texImage, x + i, y + j); // Accesso texture 2D ottimizzato
                sum += val * d_kernelConst[(j + r) * kSize + (i + r)]; // Accesso kernel da memoria costante
            }
        }
        output[y * width + x] = sum;
    }
}

// Funzione principale di elaborazione su GPU
bool processImageCUDA(const std::vector<cv::Mat>& inputChannels, 
                      std::vector<cv::Mat>& outputChannels,
                      const FilterKernel& __restrict__ filter) {
    int width = inputChannels[0].cols;
    int height = inputChannels[0].rows;
    int kSize = filter.getSize();
    const auto& kernelData = filter.getKernelData();

    outputChannels.resize(3);

    // Copia il kernel nella memoria costante (più veloce della memoria globale)
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernelConst, kernelData.data(), kSize * kSize * sizeof(float)));

    // Creazione di 3 stream CUDA per elaborazione parallela dei canali
    cudaStream_t streams[3];
    for (int i = 0; i < 3; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));


    // Allocazioni necessarie: output GPU, array CUDA (texture), oggetti texture, memoria host pinned
    std::vector<float*> d_outputs(3);
    std::vector<cudaArray*> cuArrays(3);
    std::vector<cudaTextureObject_t> texImages(3);
    std::vector<float*> h_inputs(3);
    std::vector<float*> h_outputs(3);

    for (int c = 0; c < 3; ++c) {
        // Allocazione memoria pinned per input/output: permette trasferimenti asincroni più rapidi
        CUDA_CHECK(cudaHostAlloc(&h_inputs[c], width * height * sizeof(float), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_outputs[c], width * height * sizeof(float), cudaHostAllocDefault));        

        // Conversione immagine OpenCV in buffer float (scala di grigi)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                h_inputs[c][y * width + x] = inputChannels[c].at<uchar>(y, x);

        CUDA_CHECK(cudaMalloc(&d_outputs[c], width * height * sizeof(float)));

        // Caricamento in CUDA array e creazione texture object
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        CUDA_CHECK(cudaMallocArray(&cuArrays[c], &channelDesc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArrayAsync(cuArrays[c], 0, 0, h_inputs[c], width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, streams[c]));

        struct cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArrays[c];

        struct cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        CUDA_CHECK(cudaCreateTextureObject(&texImages[c], &resDesc, &texDesc, nullptr));

        // Lancio del kernel in stream indipendente
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        filterWithTexture<<<grid, block, 0, streams[c]>>>(texImages[c], d_outputs[c], width, height, kSize);
    }

    for (int c = 0; c < 3; ++c) {
        // Copia asincrona del risultato dalla GPU alla memoria host pinned
        CUDA_CHECK(cudaMemcpyAsync(h_outputs[c], d_outputs[c], width * height * sizeof(float), cudaMemcpyDeviceToHost, streams[c]));
        CUDA_CHECK(cudaStreamSynchronize(streams[c]));

        // Ricostruzione dell'immagine OpenCV
        cv::Mat out(height, width, CV_8UC1);
        for (int i = 0; i < width * height; ++i)
            out.data[i] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, h_outputs[c][i])));
        outputChannels[c] = out;
    }

    // Cleanup delle risorse allocate
    for (int c = 0; c < 3; ++c) {
        cudaDestroyTextureObject(texImages[c]);
        cudaFreeArray(cuArrays[c]);
        cudaFree(d_outputs[c]);
        cudaFreeHost(h_inputs[c]);
        cudaFreeHost(h_outputs[c]);
        cudaStreamDestroy(streams[c]);
    }

    return true;
}
