#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>

#define TYPE double 
#define LEARNING_RATE 1e-3

#define NUM_LAYERS 5
#define TRAINING_CYCLES 10000
#define NUM_NEURONS 30

int main() {
    int nin = 3; // Number of input features
    int nout = 2; // Number of output neurons
    int nlayers = NUM_LAYERS; // Total number of layers

    srand(42); // Seed for reproducibility
    NN* nn = createNN(nin, nout, nlayers, NUM_NEURONS);

    // create some inputs and outputs
    TYPE inputs[3][3] = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    TYPE outputs[3][2] = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6}
    };

    for (int i = 0; i < TRAINING_CYCLES; i++) {
        // calculate the current loss
        TYPE total_loss = 0.0;
        for (int j = 0; j < 3; j++) {
            TYPE* output = callNN(nn, inputs[j]);
            for (int k = 0; k < nout; k++) {
                TYPE error = outputs[j][k] - output[k];
                total_loss += error * error; // Mean Squared Error
            }
        }
        printf("Cycle %d: Loss = %f\n", i, total_loss / 3.0);

        // Reset gradients
        reset_grad(nn);

        // Calculate gradients
        TYPE* inputs_ptrs[3];
        for (int j = 0; j < 3; j++) {
            inputs_ptrs[j] = inputs[j];
        }
        TYPE* outputs_ptrs[3];
        for (int j = 0; j < 3; j++) {
            outputs_ptrs[j] = outputs[j];
        }

        calculate_grad(nn, inputs_ptrs, outputs_ptrs, 3);

        // Update weights and biases
        optimise_parameters(nn, LEARNING_RATE);
    }

    return 0;
}

