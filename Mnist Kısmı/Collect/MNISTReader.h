#pragma once
#include <fstream>
#include <string>
#include <stdexcept>
#include "Network.h"

// MNIST IDX format helper functions
inline int ReverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Load MNIST images from IDX file
inline bool LoadMNISTImages(const char* filename, float*& images, int& numImages, int& rows, int& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    if (magic_number != 2051) {
        file.close();
        return false;
    }

    file.read((char*)&numImages, sizeof(numImages));
    numImages = ReverseInt(numImages);
    file.read((char*)&rows, sizeof(rows));
    rows = ReverseInt(rows);
    file.read((char*)&cols, sizeof(cols));
    cols = ReverseInt(cols);

    int imageSize = rows * cols;
    images = new float[numImages * imageSize];

    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            // 🔥 DÜZELTME: Normalize ET! (0-255 -> 0.0-1.0)
            images[i * imageSize + j] = (float)pixel / 255.0f;
        }
    }

    file.close();
    return true;
}


// Load MNIST labels from IDX file
inline bool LoadMNISTLabels(const char* filename, int*& labels, int& numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    int magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    if (magic_number != 2049) {
        file.close();
        return false;
    }

    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = ReverseInt(numLabels);

    labels = new int[numLabels];

    for (int i = 0; i < numLabels; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = (int)label;
    }

    file.close();
    return true;
}


// Save model weights to binary file
inline bool SaveModelWeights(const char* filename, DynamicMultiLayerWeights* weights) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    // Save architecture
    file.write((char*)&weights->numLayers, sizeof(int));
    int numLayerSizes = weights->numLayers + 1;
    file.write((char*)&numLayerSizes, sizeof(int));
    for (int i = 0; i < numLayerSizes; i++) {
        file.write((char*)&weights->layerSizes[i], sizeof(int));
    }

    // Save weights and biases
    for (int l = 0; l < weights->numLayers; l++) {
        int inputSize = weights->layerSizes[l];
        int outputSize = weights->layerSizes[l + 1];

        int weightCount = outputSize * inputSize;
        file.write((char*)weights->W[l], sizeof(float) * weightCount);
        file.write((char*)weights->b[l], sizeof(float) * outputSize);
    }

    file.close();
    return true;
}

// Load model weights from binary file
inline bool LoadModelWeights(const char* filename, DynamicMultiLayerWeights*& weights) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    int numLayers, numLayerSizes;
    file.read((char*)&numLayers, sizeof(int));
    file.read((char*)&numLayerSizes, sizeof(int));

    std::vector<int> layerSizes;
    for (int i = 0; i < numLayerSizes; i++) {
        int size;
        file.read((char*)&size, sizeof(int));
        layerSizes.push_back(size);
    }

    // Create architecture
    int inputDim = layerSizes[0];
    int outputDim = layerSizes[numLayerSizes - 1];
    std::vector<int> hiddenLayers;
    for (int i = 1; i < numLayerSizes - 1; i++) {
        hiddenLayers.push_back(layerSizes[i]);
    }

    weights = new DynamicMultiLayerWeights(inputDim, hiddenLayers, outputDim);

    // Load weights and biases
    for (int l = 0; l < weights->numLayers; l++) {
        int inputSize = weights->layerSizes[l];
        int outputSize = weights->layerSizes[l + 1];

        int weightCount = outputSize * inputSize;
        file.read((char*)weights->W[l], sizeof(float) * weightCount);
        file.read((char*)weights->b[l], sizeof(float) * outputSize);
    }

    file.close();
    return true;
}

// Save normalization parameters
inline bool SaveNormalizationParams(const char* filename, float* mean, float* std, int dim) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file.write((char*)&dim, sizeof(int));
    file.write((char*)mean, sizeof(float) * dim);
    file.write((char*)std, sizeof(float) * dim);

    file.close();
    return true;
}

// Load normalization parameters
inline bool LoadNormalizationParams(const char* filename, float*& mean, float*& std, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    file.read((char*)&dim, sizeof(int));
    mean = new float[dim];
    std = new float[dim];
    file.read((char*)mean, sizeof(float) * dim);
    file.read((char*)std, sizeof(float) * dim);

    file.close();
    return true;
}