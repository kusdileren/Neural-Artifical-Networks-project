#pragma once
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>

inline float Tanh(float x) {
    return std::tanh(x);
}

inline float TanhDerivative(float x) {
    float t = std::tanh(x);
    return 1.0f - t * t;
}

struct DynamicMultiLayerWeights {
    int numLayers;
    std::vector<int> layerSizes;
    std::vector<float*> W;
    std::vector<float*> b;

    DynamicMultiLayerWeights(int inputDim, std::vector<int> hiddenLayers, int outputDim) {
        numLayers = static_cast<int>(hiddenLayers.size()) + 1;

        layerSizes.push_back(inputDim);
        for (int h : hiddenLayers) {
            layerSizes.push_back(h);
        }
        layerSizes.push_back(outputDim);

        for (int l = 0; l < numLayers; l++) {
            int inputSize = layerSizes[l];
            int outputSize = layerSizes[l + 1];

            float* weights = new float[outputSize * inputSize];
            float* biases = new float[outputSize];

            float limit = std::sqrt(6.0f / (inputSize + outputSize));
            for (int i = 0; i < outputSize * inputSize; i++) {
                weights[i] = limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
            }
            for (int i = 0; i < outputSize; i++) {
                biases[i] = 0.0f;
            }

            W.push_back(weights);
            b.push_back(biases);
        }
    }

    ~DynamicMultiLayerWeights() {
        for (float* w : W) delete[] w;
        for (float* bias : b) delete[] bias;
    }
};

struct DynamicMultiLayerConfig {
    int inputDim;
    std::vector<int> hiddenLayers;
    int outputDim;
};

struct DynamicMomentumBuffers {
    std::vector<float*> velocity_W;
    std::vector<float*> velocity_b;
    float momentum_coefficient;

    DynamicMomentumBuffers(DynamicMultiLayerWeights* weights, float momentum = 0.9f) {
        momentum_coefficient = momentum;

        for (int l = 0; l < weights->numLayers; l++) {
            int inputSize = weights->layerSizes[l];
            int outputSize = weights->layerSizes[l + 1];

            float* vel_w = new float[outputSize * inputSize];
            float* vel_b = new float[outputSize];

            for (int i = 0; i < outputSize * inputSize; i++) vel_w[i] = 0.0f;
            for (int i = 0; i < outputSize; i++) vel_b[i] = 0.0f;

            velocity_W.push_back(vel_w);
            velocity_b.push_back(vel_b);
        }
    }

    ~DynamicMomentumBuffers() {
        for (float* v : velocity_W) delete[] v;
        for (float* v : velocity_b) delete[] v;
    }
};

typedef std::function<void(int epoch, float avgError)> TrainingCallback;

inline float TrainMultiLayer(
    float* Samples, float* targets, int numSample,
    DynamicMultiLayerWeights* weights,
    float learningRate,
    float* mean, float* std,
    DynamicMomentumBuffers* momentumBuf,
    float momentumValue,
    std::vector<float*>& activations,
    std::vector<float*>& derivatives,
    std::vector<float*>& deltas
)
{
    int inputDim = weights->layerSizes[0];
    int outputDim = weights->layerSizes[weights->numLayers];
    int numLayers = weights->numLayers;

    float total_err = 0.0f;

    for (int step = 0; step < numSample; step++) {

        for (int i = 0; i < inputDim; i++) {
            float std_val = (std[i] < 1e-8f) ? 1.0f : std[i];
            activations[0][i] = (Samples[step * inputDim + i] - mean[i]) / std_val;
        }

        for (int l = 0; l < numLayers; l++) {
            int inputSize = weights->layerSizes[l];
            int outputSize = weights->layerSizes[l + 1];

            for (int j = 0; j < outputSize; j++) {
                float net = weights->b[l][j];
                for (int i = 0; i < inputSize; i++) {
                    net += weights->W[l][j * inputSize + i] * activations[l][i];
                }
                activations[l + 1][j] = Tanh(net);
                derivatives[l][j] = TanhDerivative(net);
            }
        }

        int lastLayer = numLayers - 1;
        for (int j = 0; j < outputDim; j++) {

            float desired = (targets[step] == (float)j) ? 1.0f : -1.0f;

            float error = desired - activations[numLayers][j];
            deltas[lastLayer][j] = error * derivatives[lastLayer][j];
            total_err += std::fabs(error);
        }

        for (int l = numLayers - 2; l >= 0; l--) {
            int currentSize = weights->layerSizes[l + 1];
            int nextSize = weights->layerSizes[l + 2];

            for (int j = 0; j < currentSize; j++) {
                float error_sum = 0.0f;
                for (int k = 0; k < nextSize; k++) {
                    error_sum += deltas[l + 1][k] * weights->W[l + 1][k * currentSize + j];
                }
                deltas[l][j] = error_sum * derivatives[l][j];
            }
        }

        for (int l = 0; l < numLayers; l++) {
            int inputSize = weights->layerSizes[l];
            int outputSize = weights->layerSizes[l + 1];

            for (int j = 0; j < outputSize; j++) {
                for (int i = 0; i < inputSize; i++) {
                    int idx = j * inputSize + i;
                    float gradient = deltas[l][j] * activations[l][i];

                    if (momentumBuf) {
                        momentumBuf->velocity_W[l][idx] =
                            momentumValue * momentumBuf->velocity_W[l][idx] +
                            learningRate * gradient;
                        weights->W[l][idx] += momentumBuf->velocity_W[l][idx];
                    }
                    else {
                        weights->W[l][idx] += learningRate * gradient;
                    }
                }

                if (momentumBuf) {
                    momentumBuf->velocity_b[l][j] =
                        momentumValue * momentumBuf->velocity_b[l][j] +
                        learningRate * deltas[l][j];
                    weights->b[l][j] += momentumBuf->velocity_b[l][j];
                }
                else {
                    weights->b[l][j] += learningRate * deltas[l][j];
                }
            }
        }
    }

    return total_err / numSample;
}
inline int TestDynamicMultiLayer(
    float* x,
    DynamicMultiLayerWeights* weights,
    DynamicMultiLayerConfig config,
    float* mean, float* std)
{
    int inputDim = config.inputDim;
    int outputDim = config.outputDim;
    int numLayers = weights->numLayers;

    std::vector<float*> activations;
    activations.push_back(new float[inputDim]);

    for (int l = 0; l < numLayers; l++) {
        int size = weights->layerSizes[l + 1];
        activations.push_back(new float[size]);
    }

    for (int i = 0; i < inputDim; i++) {
        float std_val = (std[i] < 1e-8f) ? 1.0f : std[i];
        activations[0][i] = (x[i] - mean[i]) / std_val;
    }

    for (int l = 0; l < numLayers; l++) {
        int inputSize = weights->layerSizes[l];
        int outputSize = weights->layerSizes[l + 1];

        for (int j = 0; j < outputSize; j++) {
            float net = 0.0f;
            for (int i = 0; i < inputSize; i++) {
                net += weights->W[l][j * inputSize + i] * activations[l][i];
            }
            net += weights->b[l][j];
            activations[l + 1][j] = Tanh(net);
        }
    }

    int maxIndex = 0;
    if (outputDim > 1) {
        float maxVal = activations[numLayers][0];
        for (int i = 1; i < outputDim; i++) {
            if (activations[numLayers][i] > maxVal) {
                maxVal = activations[numLayers][i];
                maxIndex = i;
            }
        }
    }
    else {
        maxIndex = (activations[numLayers][0] > 0.0f) ? 0 : 1;
    }

    for (float* a : activations) delete[] a;

    return maxIndex;
}