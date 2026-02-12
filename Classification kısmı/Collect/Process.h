#pragma once
float* Add_Data(float* sample, int Size, float* x, int Dim);
float* Add_Labels(float* Labels, int Size, int label);
float* init_array_random(int len);
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std);
int Test_Forward(float* x, float* weight, float* bias, int neuron_count, int inputDim);

// Multi Layer için ek fonksiyonlar
struct MultiLayerWeights;
struct MultiLayerConfig;

int Test_MultiLayer_Forward(
    float* x,
    MultiLayerWeights* weights,
    MultiLayerConfig* config,
    float* mean,
    float* std
);