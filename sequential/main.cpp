#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

namespace fs = std::filesystem;

int main() {
    fs::path inputFolder  = "input";   // Cartella di input con le immagini originali
    fs::path outputFolder = "output";  // Cartella di output per salvare i risultati

    // Verifica che la cartella di input esista
    if (!fs::exists(inputFolder) || !fs::is_directory(inputFolder)) {
        std::cerr << "Cartella di input mancante: " << inputFolder << "\n";
        return 1;
    }

    // Crea la cartella di output se non esiste
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    // Definizione dei filtri da applicare
    struct FilterDef { std::string name; FilterKernel kernel; };
    std::vector<FilterDef> filters = {
        { "identity",         [](){ FilterKernel k; k.createIdentityFilter();      return k; }() },
        { "box_blur_11",      [](){ FilterKernel k; k.createBoxBlurFilter(11);     return k; }() },
        { "gaussian_15_5.0",  [](){ FilterKernel k; k.createGaussianFilter(15, 5.0f);  return k; }() },
        { "sharpen_x2",       [](){ FilterKernel k; k.createSharpenFilter(2.0f);   return k; }() },
        { "edge",             [](){ FilterKernel k; k.createEdgeFilter();          return k; }() },
        { "emboss",           [](){ FilterKernel k; k.createEmbossFilter();        return k; }() }
    };

    // Ciclo su tutti i file nella cartella di input
    for (auto const& entry : fs::directory_iterator(inputFolder)) {
        if (!entry.is_regular_file()) continue;
        auto inPath = entry.path();

        // Caricamento immagine a colori
        cv::Mat imgBGR = cv::imread(inPath.string(), cv::IMREAD_COLOR);
        if (imgBGR.empty()) {
            std::cerr << "File non valido, verrà saltato: " << inPath << "\n";
            continue;
        }

        // Divisione dell'immagine nei 3 canali B, G, R
        std::vector<cv::Mat> channels(3);
        cv::split(imgBGR, channels);

        // Ottiene nome del file senza estensione
        std::string stem = inPath.stem().string();
        std::string ext  = inPath.extension().string();

        // Applica ogni filtro definito
        for (auto& fd : filters) {
            std::vector<cv::Mat> outCh(3);  // Canali filtrati in output

            // Applica il filtro a ciascun canale separatamente
            for (int c = 0; c < 3; ++c) {
                ImageProcessor proc;

                // Converte il canale in vettore di float
                auto& mat = channels[c];
                std::vector<float> data;
                data.reserve(mat.rows * mat.cols);
                for (int y = 0; y < mat.rows; ++y)
                    for (int x = 0; x < mat.cols; ++x)
                        data.push_back(mat.at<uchar>(y, x));
                proc.setImageData(data, mat.cols, mat.rows);

                // Applica il filtro selezionato
                ImageProcessor resultProc;
                proc.applyFilterSequential(resultProc, fd.kernel);

                // Ottiene il risultato filtrato
                auto outData = resultProc.getImageData();
                cv::Mat outMat(mat.rows, mat.cols, CV_8UC1);

                // Trova valori minimi e massimi per normalizzare l'immagine
                float minVal = outData.front(), maxVal = outData.front();
                for (float v : outData) {
                    minVal = std::min(minVal, v);
                    maxVal = std::max(maxVal, v);
                }
                float range = maxVal - minVal;

                // Normalizza ogni valore su scala [0,255]
                for (int y = 0; y < outMat.rows; ++y) {
                    for (int x = 0; x < outMat.cols; ++x) {
                        float v = outData[y * outMat.cols + x];
                        float norm = (range > 1e-5f) ? ((v - minVal) * 255.0f / range) : 127.0f;
                        outMat.at<uchar>(y, x) = static_cast<uchar>(std::clamp(int(std::round(norm)), 0, 255));
                    }
                }

                // Salva il canale filtrato
                outCh[c] = outMat;
            }

            // Unisce i canali filtrati e salva l'immagine risultante
            cv::Mat outBGR;
            cv::merge(outCh, outBGR);
            fs::path outPath = outputFolder / (stem + "_" + fd.name + ext);
            if (!cv::imwrite(outPath.string(), outBGR)) {
                std::cerr << "Errore nel salvataggio: " << outPath << "\n";
            } else {
                std::cout << "Immagine elaborata: " << inPath.filename()
                          << " -> " << outPath.filename() << "\n";
            }
        }
    }

    std::cout << "Tutte le immagini sono state elaborate.\n";
    return 0;
}
