#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TYPE double 

typedef struct Neuron {
    TYPE* weights;
    int num_weights;
    TYPE bias;
    TYPE value;

    TYPE bias_grad; // NOT MEMORY EFFICIENT WHEN DOING INFERENCE
    TYPE* weights_grad; // Gradient for each weight // NOT MEMORY EFFICIENT WHEN DOING INFERENCE
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
        nn->layers[i].num_neurons = (i == nlayers - 1) ? nout : 10; // Example: 10 neurons in hidden layers
        nn->layers[i].neurons = malloc(nn->layers[i].num_neurons * sizeof(Neuron));
        
        for (int j = 0; j < nn->layers[i].num_neurons; j++) {
            int weights_size = (i == 0) ? nin : nn->layers[i - 1].num_neurons;

            nn->layers[i].neurons[j].weights = malloc(weights_size * sizeof(TYPE));
            nn->layers[i].neurons[j].bias = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0); // Random bias between -1 and 1
            nn->layers[i].neurons[j].num_weights = weights_size;

            for (int k = 0; k < weights_size; k++) {
                nn->layers[i].neurons[j].weights[k] = ((TYPE)rand() / (TYPE)RAND_MAX * (TYPE)2.0 - (TYPE)1.0); // Random weight between -1 and 1
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
                neuron->value = (neuron->value > 0) ? neuron->value : 0;
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
                printf("%f", nn->layers[i].neurons[j].weights[k]);
            }
            printf("]\n");
        }
    }
}

int main() {
    int nin = 3; // Number of input features
    int nout = 2; // Number of output neurons
    int nlayers = 3; // Total number of layers

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

    for (int i = 0; i < 100; i++) { // 100 training cycles
        // calculate loss
        TYPE total_loss = 0.0;
        for (int j = 0; j < 3; j++) {
            TYPE* output = callNN(nn, inputs[j]);
            for (int k = 0; k < nout; k++) {
                TYPE loss = outputs[j][k] - output[k];
                loss *= loss; // Squared loss
                total_loss += loss;
                // The plan. it's not really orthodox, but it could work.
                // 0. Reset gradients BEFORE this current loop starts.
                // 1. In HERE, in the current loop of the input, we do the backpropagation and we add the gradients to each weights and biases,
                // 2. After the loop ends, we will have all the gradients accumulated for each weight and bias.
                // 3. After the loop ends, we will update the weights and biases using the accumulated gradients. We will do -0.01 * gradient / outputs_sample_size, because we don't want to overupdate when we have more samples.
            }
        }

        printf("Epoch %d, Loss: %f\n", i, total_loss);
    }

    return 0;
}

