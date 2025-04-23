# Sequential CPU Implementation using C++ and OpenCV

This project performs RGB image processing using various convolution-based filters. It is written in C++ and uses OpenCV for image I/O and file management.

---

## Filters Implemented

All filters are based on standard convolution kernels. The following filters are included:

- `identity` 
- `Uniform blur`
- `Gaussian blur`
- `Sharpening filter`
- `Edge detection`
- `Emboss effect`

The filter generation is dynamic and scalable, especially useful for high-resolution images.

---

## Structure

- `main.cpp` — Loads input images, applies filters, saves output
- `image_processing.h/cpp` — Contains the classes:
  - `FilterKernel`: creates and stores convolution matrices
  - `ImageProcessor`: handles filter application with padding

---

## How It Works
- Reads each image from the input folder.
- Splits the image into 3 channels (B, G, R).
- Applies each filter to every channel.
- Normalizes filtered data and recombines the channels.
- Saves processed images to the output folder.
