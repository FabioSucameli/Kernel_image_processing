#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

// Funzione per misurare il tempo di esecuzione
template<typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main() {
    fs::path inputFolder = "input";
    fs::path outputFolder = "output";

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
        { "identity",        [](){ FilterKernel k; k.createIdentityFilter();    return k; }() },
        { "box_blur_11",     [](){ FilterKernel k; k.createBoxBlurFilter(11);   return k; }() },
        { "gaussian_15_5.0", [](){ FilterKernel k; k.createGaussianFilter(15, 5.0f); return k; }() },
        { "sharpen_x2",      [](){ FilterKernel k; k.createSharpenFilter(2.0f); return k; }() },
        { "edge",            [](){ FilterKernel k; k.createEdgeFilter();        return k; }() },
        { "emboss",          [](){ FilterKernel k; k.createEmbossFilter();      return k; }() }
    };

    // Ciclo su tutti i file nella cartella di input
    for (auto const& entry : fs::directory_iterator(inputFolder)) {
        if (!entry.is_regular_file()) continue;
        auto inPath = entry.path();

        // Carica immagine a colori
        cv::Mat imgBGR = cv::imread(inPath.string(), cv::IMREAD_COLOR);
        if (imgBGR.empty()) {
            std::cerr << "File non valido, verrà saltato: " << inPath << "\n";
            continue;
        }

        // Divisione nei canali B, G, R
        std::vector<cv::Mat> channels(3);
        cv::split(imgBGR, channels);

        // Ottiene il nome base del file
        std::string stem = inPath.stem().string();
        std::string ext = inPath.extension().string();

        // Applica ogni filtro definito
        for (auto& fd : filters) {
            std::vector<cv::Mat> outCh(3);
            double timeGPU = 0.0, timeCPU = 0.0;
            //Misura tempo GPU
            timeGPU = measureTime([&]() {
                processImageCUDA(channels, outCh, fd.kernel);
            });

            // Misura anche tempo CPU
            timeCPU = measureTime([&]() {
                for (int c = 0; c < 3; ++c) {
                    ImageProcessor proc;
                    auto& mat = channels[c];
                    std::vector<float> data;
                    data.reserve(mat.rows * mat.cols);
                    for (int y = 0; y < mat.rows; ++y)
                        for (int x = 0; x < mat.cols; ++x)
                            data.push_back(mat.at<uchar>(y, x));
                    proc.setImageData(data, mat.cols, mat.rows);

                    ImageProcessor resultProc;
                    proc.applyFilterSequential(resultProc, fd.kernel);
                }
            });

            // Unisce i canali filtrati e salva il risultato
            cv::Mat outBGR;
            cv::merge(outCh, outBGR);
            fs::path outPath = outputFolder / (stem + "_" + fd.name + ext);
            if (!cv::imwrite(outPath.string(), outBGR)) {
                std::cerr << "Errore nel salvataggio: " << outPath << "\n";
            } else {
                std::cout << "Immagine elaborata: " << inPath.filename()
                          << " -> " << outPath.filename()
                          << " [GPU: " << timeGPU << "ms, CPU: " << timeCPU << "ms, Speedup: "
                          << (timeCPU > 0 ? timeCPU / timeGPU : 0) << "x]\n";
            }
        }
    }

    std::cout << "Tutte le immagini sono state elaborate.\n";
    return 0;
}
