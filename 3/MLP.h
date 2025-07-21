#ifndef MLP_H 
#define MLP_H 

#define TYPE double 

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


NN* createNN(int nin, int nout, int nlayers, int num_neurons);

TYPE* callNN(NN* nn, TYPE* inputs);

int reset_grad(NN* nn);

int calculate_grad(NN* nn, TYPE* inputs[], TYPE* outputs[], int samples_count);

int optimise_parameters(NN* nn, TYPE learning_rate);

void visualiseNN(NN* nn);

#endif 

