## CUDA Parallelization

This project implements GPU-accelerated image filtering using CUDA. The parallelization targets the convolution step, which is computationally intensive, especially for large images and complex filters.

### How It Works

The CUDA implementation performs 2D convolution. Each color channel (B, G, R) is processed in parallel using separate CUDA streams.

The core of the parallelization is the `filterWithTexture` kernel. Each thread computes a single output pixel by:
- Accessing the neighborhood pixels from the texture memory
- Multiplying them by the corresponding kernel values stored in constant memory
- Writing the result into a linear output buffer

Finally, the filtered data is copied back to the host asynchronously and converted back into OpenCV images.

### Hardware

All experiments were run on a PC equipped with an **NVIDIA GeForce RTX 4060**, which provides high throughput and an efficient memory hierarchy suitable for high-resolution image processing.

### Usage

To compile and run the project, use the following command (Windows with OpenCV and CUDA):

```bash
nvcc image_processing.cu image_processing.cpp main.cpp -o image_filter.exe ^
  -I"C:\opencv\build\include" ^
  -L"C:\opencv\build\x64\vc16\lib" -lopencv_world4110 ^
  -std=c++17 -arch=sm_89
