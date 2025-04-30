#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Classe FilterKernel: Rappresenta un kernel per applicare filtri all'immagine
class FilterKernel {
public:
    FilterKernel();              // Costruttore
    ~FilterKernel();             // Distruttore

    void displayKernel() const; // Stampa la matrice del kernel a schermo (debug)

    // Crea diversi tipi di kernel
    bool createIdentityFilter();                          // Filtro identità (nessun cambiamento)
    bool createBoxBlurFilter(int size = 3);               // Filtro di media (blur) con dimensione specifica
    bool createGaussianFilter(int size, float sigma);     // Filtro Gaussiano parametrico
    bool createSharpenFilter(float factor = 1.0f);        // Filtro di sharpening con intensità personalizzata
    bool createEdgeFilter();                              // Filtro per rilevamento bordi (laplaciano)
    bool createEmbossFilter();                            // Filtro per effetto rilievo

    int getSize() const;                                  // Ritorna la dimensione
    std::vector<float> getKernelData() const;             // Ritorna il vettore con i valori del kernel

private:
    std::vector<float> kernelData;  // Dati della matrice del filtro
    int size;                       // Dimensione della matrice
};


// Classe ImageProcessor: rappresenta e gestisce una singola immagine in scala di grigi
//  Applica il filtro specificato usando convoluzione
class ImageProcessor {
public:
    ImageProcessor();               // Costruttore
    ~ImageProcessor();              // Distruttore

    // Carica i dati immagine come vettore di float e imposta dimensione
    bool setImageData(const std::vector<float>& data, int width, int height);

    std::vector<float> getImageData() const;  // Ritorna i dati filtrati

    // Applica un filtro in modalità sequenziale 
    bool applyFilterSequential(ImageProcessor& output, const FilterKernel& filter) const;
    

private:
    // Applica il filtro e ritorna direttamente i dati risultanti
    std::vector<float> applyFilterCore(const FilterKernel& filter) const;

    // Crea una nuova immagine con bordo aggiuntivo (padding) per gestire i contorni
    std::vector<float> createPaddedImage(int padY, int padX) const;

    std::vector<float> imageData;  // Dati dei pixel dell'immagine
    int width;                     // Larghezza dell'immagine
    int height;                    // Altezza dell'immagine
};

// Funzione per elaborare tutti i canali di un'immagine in parallelo
bool processImageCUDA(const std::vector<cv::Mat>& inputChannels, 
                     std::vector<cv::Mat>& outputChannels,
                     const FilterKernel& filter);

#endif