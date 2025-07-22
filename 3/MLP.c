#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MLP.h"

#define TYPE double 

NN* createNN(int nin, int nout, int nlayers, int num_neurons) {
    NN* nn = malloc(sizeof(NN));

    nn->num_layers = nlayers;
    nn->layers = malloc(nlayers * sizeof(Layer));

    for (int i = 0; i < nlayers; i++) {
        nn->layers[i].num_neurons = (i == nlayers - 1) ? nout : num_neurons; // Example: 10 neurons in hidden layers
        nn->layers[i].neurons = malloc(nn->layers[i].num_neurons * sizeof(Neuron));
        
        for (int j = 0; j < nn->layers[i].num_neurons; j++) {
            int weights_size = (i == 0) ? nin : nn->layers[i - 1].num_neurons;

            nn->layers[i].neurons[j].weights = malloc(weights_size * sizeof(TYPE));

            TYPE random_bias = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0); // Random bias between -1 and 1
            if(i == nlayers - 1) {
                // tanh
                nn->layers[i].neurons[j].bias = random_bias * sqrt(1.0 / (TYPE)weights_size); // Scale bias for output layer
            } else {
                // ReLU
                nn->layers[i].neurons[j].bias = random_bias * sqrt(2.0 / (TYPE)weights_size); // Scale bias for hidden layers
            }

            nn->layers[i].neurons[j].num_weights = weights_size;

            nn->layers[i].neurons[j].value_grad = 0.0; // Initialize gradient for value
            nn->layers[i].neurons[j].weights_grad = malloc(weights_size * sizeof(TYPE)); // Initialize gradient for weights
            nn->layers[i].neurons[j].bias_grad = 0.0; // Initialize gradient for bias

            for (int k = 0; k < weights_size; k++) {
                TYPE random_weight = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0); // Random weight between -1 and 1
                if (i == nlayers - 1) {
                    // tanh
                    nn->layers[i].neurons[j].weights[k] = random_weight * sqrt(1.0 / (TYPE)weights_size); // Scale weights for output layer
                } else {
                    // ReLU
                    nn->layers[i].neurons[j].weights[k] = random_weight * sqrt(2.0 / (TYPE)weights_size); // Scale weights for hidden layers
                }

                nn->layers[i].neurons[j].weights_grad[k] = 0.0; // Initialize weight gradient
            }
        }
    }
    return nn;
}

TYPE* callNN(NN* nn, TYPE* inputs) {
    nn->inputs = inputs;

    for (int i = 0; i < nn->num_layers; i++) {
        for (int j = 0; j < nn->layers[i].num_neurons; j++) {
            Neuron* neuron = &nn->layers[i].neurons[j];
            neuron->value = neuron->bias; // Start with bias

            for (int k = 0; k < neuron->num_weights; k++) {
                if (i == 0) {
                    // Input layer
                    neuron->value += inputs[k] * neuron->weights[k];
                } else {
                    // Hidden and output layers
                    neuron->value += nn->layers[i - 1].neurons[k].value * neuron->weights[k];
                }
            }

            // Apply activation function (ReLu if hidden, tanh if output)
            if (i < nn->num_layers - 1) {
                // ReLU activation for hidden layers
                // don't do relu, instead use the weird relu taht doesn't get to 0
                neuron->value = (neuron->value > 0) ? neuron->value : 0.01 * neuron->value; // Leaky ReLU
            } else {
                // tanh activation for output layer
                neuron->value = tanh(neuron->value); // IMPORTANT: tanh function is used here, tanh function accepts DOUBLE values, if I change the TYPE to float, it might not work correctly...
            }
        }
    }

    // Collect outputs from the last layer
    TYPE* outputs = malloc(nn->layers[nn->num_layers - 1].num_neurons * sizeof(TYPE));
    for (int j = 0; j < nn->layers[nn->num_layers - 1].num_neurons; j++) {
        outputs[j] = nn->layers[nn->num_layers - 1].neurons[j].value;
    }

    return outputs;
}

void visualiseNN(NN* nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        printf("Layer %d:\n", i);
        for (int j = 0; j < nn->layers[i].num_neurons; j++) {
            printf("[");
            for(int k = 0; k < nn->layers[i].neurons[j].num_weights; k++) {
                if (k > 0) printf(", ");
                printf("%f (%f)", nn->layers[i].neurons[j].weights[k], nn->layers[i].neurons[j].weights_grad[k]);
            }
            printf("] %f (%f)\n", nn->layers[i].neurons[j].bias, nn->layers[i].neurons[j].bias_grad);
        }
    }
}

int reset_grad(NN* nn) {
    for (int l = 0; l < nn->num_layers; l++) {
        for (int m = 0; m < nn->layers[l].num_neurons; m++) {
            nn->layers[l].neurons[m].value_grad = 0.0;
            nn->layers[l].neurons[m].bias_grad = 0.0;
            for (int k = 0; k < nn->layers[l].neurons[m].num_weights; k++) {
                nn->layers[l].neurons[m].weights_grad[k] = 0.0;
            }
        }
    }
    return 0;
}

int calculate_grad(NN* nn, TYPE* inputs[], TYPE* outputs[], int samples_count) {
    for (int j = 0; j < samples_count; j++) {
        TYPE* output = callNN(nn, inputs[j]);

        for(int k = nn->num_layers -1; k >= 0; k--) {
            for(int l = 0; l < nn->layers[k].num_neurons; l++) {
                Neuron* neuron = &nn->layers[k].neurons[l];

                if (k == nn->num_layers - 1) {
                    // Output layer: each neuron only for its own output!
                    TYPE error = outputs[j][l] - output[l];
                    TYPE loss = error * error;
                    TYPE derivative = -2.0 * error;
                    neuron->value_grad = derivative * (1.0 - output[l] * output[l]);
                } else {
                    TYPE sum_grad = 0.0;
                    for (int m = 0; m < nn->layers[k + 1].num_neurons; m++) {
                        sum_grad += nn->layers[k + 1].neurons[m].weights[l] * nn->layers[k + 1].neurons[m].value_grad;
                    }
                    neuron->value_grad = sum_grad * ((neuron->value > 0) ? 1.0 : 0.01); // Leaky ReLU derivative
                }
                neuron->bias_grad += neuron->value_grad; // Accumulate bias gradient

                for (int m = 0; m < neuron->num_weights; m++) {
                    if (k == 0) {
                        // Input layer
                        neuron->weights_grad[m] += neuron->value_grad * inputs[j][m];
                    } else {
                        // Hidden and output layers
                        neuron->weights_grad[m] += neuron->value_grad * nn->layers[k - 1].neurons[m].value;
                    }
                }
            }
        }
    }

    return 0;
}

int optimise_parameters(NN* nn, TYPE learning_rate, int sample_size) {
    for (int l = 0; l < nn->num_layers; l++) {
        for (int m = 0; m < nn->layers[l].num_neurons; m++) {
            Neuron* neuron = &nn->layers[l].neurons[m];
            neuron->bias -= learning_rate * neuron->bias_grad / sample_size; // Update bias

            for (int k = 0; k < neuron->num_weights; k++) {
                neuron->weights[k] -= learning_rate * neuron->weights_grad[k] / sample_size; // Update weights
            }
        }
    }

    return 0;
}

