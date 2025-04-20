#include "image_processing.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159  //Definizione di pi greco per il filtro Gaussiano
#endif

// Costruttore e distruttore
FilterKernel::FilterKernel() : size(0) {}
FilterKernel::~FilterKernel() { kernelData.clear(); }



// Filtro identità: non modifica l'immagine
bool FilterKernel::createIdentityFilter() {
    size = 3;
    kernelData = {0,0,0, 0,1,0, 0,0,0};
    return true;
}

// Filtro di media (box blur) parametrico
bool FilterKernel::createBoxBlurFilter(int kSize) {
    if (kSize % 2 == 0 || kSize < 3) return false;
    size = kSize;
    int total = size * size;
    kernelData.assign(total, 1.0f / static_cast<float>(total));
    return true;
}

// Filtro Gaussiano parametrico
bool FilterKernel::createGaussianFilter(int kSize, float sigma) {
    if (kSize % 2 == 0 || sigma <= 0.0f || kSize < 3) return false;
    size = kSize;
    kernelData.resize(size * size);
    int center = size / 2;
    float sum = 0.0f;
    // Calcolo della formula del Gaussiano
    for (int y = -center; y <= center; ++y) {
        for (int x = -center; x <= center; ++x) {
            float value = std::exp(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernelData[(y + center) * size + (x + center)] = value;
            sum += value;
        }
    }
    // Normalizzazione (somma = 1)
    for (auto& v : kernelData) v /= sum;
    return true;
}

// Filtro di sharpening (accentua i bordi), parametrico in base a factor
bool FilterKernel::createSharpenFilter(float factor) {
    size = 3;
    kernelData = {
        0, -1 * factor, 0,
        -1 * factor, 4 * factor + 1, -1 * factor,
        0, -1 * factor, 0
    };
    return true;
}

// Filtro per rilevamento contorni (laplaciano 8-neighbors)
bool FilterKernel::createEdgeFilter() {
    size = 3;
    kernelData = {-1,-1,-1, -1,8,-1, -1,-1,-1};
    return true;
}

// Filtro emboss: crea effetto rilievo
bool FilterKernel::createEmbossFilter() {
    size = 3;
    kernelData = {-2,-1,0, -1,1,1, 0,1,2};
    return true;
}

// Dimensione e dati kernel
int FilterKernel::getSize() const { return size; }
std::vector<float> FilterKernel::getKernelData() const { return kernelData; }


// Implementazione di ImageProcessor
ImageProcessor::ImageProcessor() : width(0), height(0) {}
ImageProcessor::~ImageProcessor() = default;

// Imposta immagine da un vettore di float (scala di grigi)
bool ImageProcessor::setImageData(const std::vector<float>& data, int w, int h) {
    imageData = data;
    width = w;
    height = h;
    return true;
}

// Ritorna i dati dell'immagine (vettore)
std::vector<float> ImageProcessor::getImageData() const {
    return imageData;
}

// Crea immagine con bordo a zero
std::vector<float> ImageProcessor::createPaddedImage(int padY, int padX) const {
    int newW = width + 2 * padX;
    int newH = height + 2 * padY;
    std::vector<float> padded(newW * newH, 0.0f);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            padded[(y + padY) * newW + (x + padX)] = imageData[y * width + x];
    return padded;
}

// Applica il filtro di convoluzione e restituisce una nuova immagine filtrata
std::vector<float> ImageProcessor::applyFilterCore(const FilterKernel& filter) const {
    int k = filter.getSize();       // dimensione kernel
    int r = k / 2;                  // raggio
    auto kernel = filter.getKernelData();
    auto pad = createPaddedImage(r, r); // immagine con padding

    std::vector<float> out(width * height, 0.0f);
    int paddedW = width + 2 * r;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;

            // Applicazione della convoluzione
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    sum += pad[(y + dy + r) * paddedW + (x + dx + r)] * kernel[(dy + r) * k + (dx + r)];
                }
            }
            out[y * width + x] = sum;
        }
    }
    return out;
}

// Applica il filtro e salva il risultato in un altro oggetto ImageProcessor
bool ImageProcessor::applyFilterSequential(ImageProcessor& output, const FilterKernel& filter) const {
    auto result = applyFilterCore(filter);
    return output.setImageData(result, width, height);
}

// Per Debugging
// Stampa il contenuto del kernel su console
void FilterKernel::displayKernel() const {
    if (size == 0 || kernelData.empty()) { std::cout << "Kernel not initialized\n"; return; }
    std::cout << "Kernel " << size << "x" << size << ":\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j)
            std::cout << std::fixed << std::setprecision(4)
                      << kernelData[i*size + j] << " ";
        std::cout << "\n";
    }
}