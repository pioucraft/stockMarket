#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TYPE double 
#define LEARNING_RATE 1e-3

#define NUM_LAYERS 5
#define NUM_NEURONS 30
#define TRAINING_CYCLES 10000


typedef struct Neuron {
    TYPE* weights;
    int num_weights;
    TYPE bias;
    TYPE value; // NOT MEMORY EFFICIENT

    TYPE value_grad; // NOT MEMORY EFFICIENT
    TYPE* weights_grad; // NOT MEMORY EFFICIENT
    TYPE bias_grad; // NOT MEMORY EFFICIENT
} Neuron;

typedef struct Layer {
    int num_neurons;
    Neuron* neurons;
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
    TYPE *inputs;
    // to access inputs size, we can use nn->layers[0].neurons[0].num_weights
    // to access outputs, we can use nn->layers[nn->num_layers - 1].neurons[i].value where i is for i = 0 to i < nn->layers[nn->num_layers - 1].num_neurons
} NN;

NN* createNN(int nin, int nout, int nlayers) {
    NN* nn = malloc(sizeof(NN));

    nn->num_layers = nlayers;
    nn->layers = malloc(nlayers * sizeof(Layer));

    for (int i = 0; i < nlayers; i++) {
        nn->layers[i].num_neurons = (i == nlayers - 1) ? nout : NUM_NEURONS; // Example: 10 neurons in hidden layers
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

int main() {
    FILE *loss_file = fopen("loss.csv", "w");
    if (!loss_file) {
        perror("Failed to open loss.csv");
        return 1;
    }


    int nin = 3; // Number of input features
    int nout = 2; // Number of output neurons
    int nlayers = NUM_LAYERS; // Total number of layers

    srand(42); // Seed for reproducibility
    NN* nn = createNN(nin, nout, nlayers);

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

    TYPE last_loss = 100000.0; // Initialize a large last loss value to ensure the first loss is always smaller
    for (int i = 0; i < TRAINING_CYCLES; i++) {
        // calculate loss
        TYPE total_loss = 0.0;
        // Reset gradients
        for (int l = 0; l < nn->num_layers; l++) {
            for (int m = 0; m < nn->layers[l].num_neurons; m++) {
                nn->layers[l].neurons[m].value_grad = 0.0;
                nn->layers[l].neurons[m].bias_grad = 0.0;
                for (int k = 0; k < nn->layers[l].neurons[m].num_weights; k++) {
                    nn->layers[l].neurons[m].weights_grad[k] = 0.0;
                }
            }
        }
        for (int j = 0; j < 3; j++) {
            TYPE* output = callNN(nn, inputs[j]);

            for(int k = nn->num_layers -1; k >= 0; k--) {
                for(int l = 0; l < nn->layers[k].num_neurons; l++) {
                    Neuron* neuron = &nn->layers[k].neurons[l];

                    if (k == nn->num_layers - 1) {
                        // Output layer: each neuron only for its own output!
                        TYPE error = outputs[j][l] - output[l];
                        TYPE loss = error * error;
                        TYPE derivative = -2.0 * error;
                        total_loss += loss;
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

        /*
        if(last_loss < total_loss) {
            printf("Loss increased from %.15f to %.15f at cycle %d, stopping training.\n", last_loss, total_loss, i);
        }
        */
        if(i % 1000 == 0) {
            printf("Cycle %d, Loss: %.15f\n", i, total_loss);
        }
        fprintf(loss_file, "%d,%.15f\n", i, total_loss);
        last_loss = total_loss; // Update last loss

        // Update weights and biases
        for (int l = 0; l < nn->num_layers; l++) {
            for (int m = 0; m < nn->layers[l].num_neurons; m++) {
                Neuron* neuron = &nn->layers[l].neurons[m];
                neuron->bias -= LEARNING_RATE * neuron->bias_grad / 3; // Update bias

                for (int k = 0; k < neuron->num_weights; k++) {
                    neuron->weights[k] -= LEARNING_RATE * neuron->weights_grad[k] / 3; // Update weights
                }
            }
        }
    }

    return 0;
}

