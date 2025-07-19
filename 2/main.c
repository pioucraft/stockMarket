#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HI printf("Hello, World!\n");

typedef struct BackpropValue {
    int id;
    float value;
    struct BackpropValue **parents;
    int num_parents;
    char operation;
    float grad;
} BackpropValue;

typedef struct Neuron {
    struct BackpropValue **w;
    int nin;
    struct BackpropValue *b;
} Neuron;

typedef struct Layer {
    struct Neuron **neurons;
    int nout;
} Layer;

typedef struct NN {
    struct Layer **layers;
    int n_layers;
    int nin;
    int nout;
    int nhin; // Number of hidden neurons in every layer of the hidden layers
} NN;

BackpropValue **parameters;
int parameters_length = 0;

int displayValueWithDepth(BackpropValue *bv, int depth) {
    printf("%i Value: %f, Parents: %p, NumParents: %d, Operation: %c, Grad : %f\n",
           bv->id, bv->value, bv->parents, bv->num_parents,
           bv->operation, bv->grad);
    if (depth > 0 && bv->num_parents > 0) {
        for (int i = 0; i < bv->num_parents; i++) {
            displayValueWithDepth(bv->parents[i], depth - 1);
        }
    }
    return 0;
}

int displayValue(BackpropValue *bv) { 
    return displayValueWithDepth(bv, 0); 
}

int addValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value + b->value;

    result->parents = malloc(sizeof(BackpropValue*) * 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->operation = '+';
    return 0;
}

int subtractValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value - b->value;

    result->parents = malloc(sizeof(BackpropValue*) * 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->operation = '-';
    return 0;
}

int multiplyValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value * b->value;

    result->parents = malloc(sizeof(BackpropValue*) * 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->num_parents = 2;
    result->operation = '*';
    return 0;
}

int tanhValue(BackpropValue *a, BackpropValue *result) {
    result->value = tanhf(a->value);

    result->parents = malloc(sizeof(BackpropValue*) * 1);
    result->parents[0] = a;
    result->num_parents = 1;
    result->operation = 'T'; // No operation
    return 0;
}

int buildTopo(BackpropValue *bv, BackpropValue ***topo, int *index) {
    for(int i = 0; i < bv->num_parents; i++) {
        int found = 0;
        for(int j = 0; j < *index; j++) {
            if ((*topo)[j] == (bv->parents[i])) {
                found = 1;
                break;
            }
        }
        if (found == 0) {
            buildTopo(bv->parents[i], topo, index);
            *topo = realloc(*topo, sizeof(BackpropValue *) * (*index + 1));
            (*topo)[*index] = (bv->parents[i]);
            (*index)++;
        }
    }
    return 0;
}
int _backwardValue(BackpropValue *bv) {
    if(bv->operation == '_') {
        return 0;
    } else if(bv->operation == '+') {
        bv->parents[0]->grad += bv->grad;
        bv->parents[1]->grad += bv->grad;
    } else if(bv->operation == '-') {
        bv->parents[0]->grad += bv->grad;
        bv->parents[1]->grad -= bv->grad;
    } else if(bv->operation == '*') {
        bv->parents[0]->grad += bv->grad * bv->parents[1]->value;
        bv->parents[1]->grad += bv->grad * bv->parents[0]->value;
    } else if(bv->operation == 'T') {
        float tanh_val = tanhf(bv->parents[0]->value);
        bv->parents[0]->grad += bv->grad * (1 - tanh_val * tanh_val);
    } else {
        return -1; // Unknown operation
    }

    return 0;
}

int resetGrad(BackpropValue *bv) {
    bv->grad = 0.0f; // Reset the gradient to 0
    for(int i = 0; i < bv->num_parents; i++) {
        resetGrad(bv->parents[i]);
    }
    return 0;
}

int backwardValue(BackpropValue *bv) {
    // start by doing topological sort.
    BackpropValue **topo = NULL;
    int *topo_length = malloc(sizeof(int));

    buildTopo(bv, &topo, topo_length);

    bv->grad = 1.0f; // Set the gradient of the output value to 1.0
    _backwardValue(bv);

    // It's -2, because we add an index++ at the end of buildTopo
    for(int i = *topo_length-2; i >= 0; i--) {
        _backwardValue(topo[i]);
    }

    free(topo);
    free(topo_length);

    return 0;
}

int createValue(float value, BackpropValue *bv) {
    static int id_counter = 0; // Static counter to assign unique IDs
    
    bv->id = id_counter++;
    bv->value = value;
    bv->parents = NULL;
    bv->num_parents = 0;
    bv->operation = '_'; // No operation
    bv->grad = 0.0f; // Initialize gradient to 0
    return 0;
}

int createNeuron(int nin, Neuron *n) {
    n->nin = nin;
    n->w = malloc(sizeof(BackpropValue*) * nin);
    for(int i = 0; i < nin; i++) {
        // random float between -1.0 and 1.0
        float random_value = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float in [-1, 1]
        n->w[i] = malloc(sizeof(BackpropValue));

        // add to parameters

        parameters_length++;
        parameters = realloc(parameters, sizeof(BackpropValue*) * parameters_length);
        parameters[parameters_length - 1] = n->w[i];
        
        createValue(random_value, n->w[i]);
    }
    float random_bias = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float in [-1, 1]
    n->b = malloc(sizeof(BackpropValue));
    
    // add to parameters
    parameters_length++;
    parameters = realloc(parameters, sizeof(BackpropValue*) * parameters_length);
    parameters[parameters_length - 1] = n->b;

    createValue(random_bias, n->b);
    return 0;
}

int callNeuron(Neuron *n, BackpropValue **inputs, BackpropValue *output) {
    BackpropValue *act = NULL;
    BackpropValue *oldAct = NULL;

    for(int i = 0; i < n->nin; i++) {
        if(act != NULL) {
            oldAct = act;
        } else {
            oldAct = malloc(sizeof(BackpropValue));
            createValue(0.0f, oldAct);
        }

        act = malloc(sizeof(BackpropValue));
        createValue(0.0f, act); // Initialize activation value
        
        BackpropValue *weighted_input = malloc(sizeof(BackpropValue));
        createValue(0.0f, weighted_input);
        multiplyValues(n->w[i], inputs[i], weighted_input);
        
        addValues(oldAct, weighted_input, act);
    }

    // Add the bias
    BackpropValue *final_act = malloc(sizeof(BackpropValue));
    createValue(0.0f, final_act); // Initialize final activation value
    addValues(act, n->b, final_act);

    // Apply activation function (tanh)
    createValue(0.0f, output); // Initialize output value
    tanhValue(final_act, output);
    return 0;
}

int createLayer(int nin, int nout, Layer *l) {
    l->nout = nout;
    l->neurons = malloc(sizeof(Neuron*) * nout);
    for(int i = 0; i < nout; i++) {
        l->neurons[i] = malloc(sizeof(Neuron));
        createNeuron(nin, l->neurons[i]);
    }
    return 0;
}

int callLayer(Layer *l, BackpropValue **inputs, BackpropValue **outputs) {
    for(int i = 0; i < l->nout; i++) {
        outputs[i] = malloc(sizeof(BackpropValue));
        createValue(0.0f, outputs[i]); // Initialize output value
        callNeuron(l->neurons[i], inputs, outputs[i]);
    }
    return 0;
}

int createNN(int nin, int nout, int n_layers, NN *nn, int nhin) {
    if (n_layers < 2) {
        fprintf(stderr, "Error: Number of layers must be at least 2, including input and output layers.\n");
        return -1;
    }

    nn->nin = nin;
    nn->nout = nout;
    nn->n_layers = n_layers;
    nn->layers = malloc(sizeof(Layer*) * n_layers);
    nn->nhin = nhin;

    nn->layers[0] = malloc(sizeof(Layer));
    createLayer(nin, nhin, nn->layers[0]); // Input layer

    for(int i = 1; i < n_layers - 1; i++) {
        nn->layers[i] = malloc(sizeof(Layer));
        createLayer(nhin, nhin, nn->layers[i]);
    }

    nn->layers[n_layers - 1] = malloc(sizeof(Layer));
    createLayer(nhin, nout, nn->layers[n_layers - 1]); // Output layer

    return 0;
}

int callNN(NN *nn, BackpropValue **inputs, BackpropValue **outputs) {
    // call the first layer
    BackpropValue **hidden_outputs = malloc(sizeof(BackpropValue*) * nn->nhin);
    callLayer(nn->layers[0], inputs, hidden_outputs);

    // call the hidden layers
    for(int i = 1; i < nn->n_layers - 1; i++) {
        BackpropValue **new_hidden_outputs = malloc(sizeof(BackpropValue*) * nn->nhin);
        callLayer(nn->layers[i], hidden_outputs, new_hidden_outputs);

        free(hidden_outputs);
        hidden_outputs = new_hidden_outputs;
    }

    // call the output layer
    callLayer(nn->layers[nn->n_layers - 1], hidden_outputs, outputs);
    free(hidden_outputs);

    return 0;
}

void dumpGraphviz(FILE *f, BackpropValue *v, int *visited, int max_nodes) {
    if (visited[v->id]) return;
    visited[v->id] = 1;

    fprintf(f, "v%d [label=\"id:%d\\nval:%.2f\\ngrad:%.2f\\nop:%c\"];\n",
            v->id, v->id, v->value, v->grad, v->operation);

    for (int i = 0; i < v->num_parents; i++) {
        fprintf(f, "v%d -> v%d;\n", v->parents[i]->id, v->id);
        dumpGraphviz(f, v->parents[i], visited, max_nodes);
    }
}

void writeDotFile(BackpropValue *output, const char *filename, int total_values) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Could not open %s for writing.\n", filename);
        return;
    }

    fprintf(f, "digraph G {\n");
    int *visited = calloc(total_values, sizeof(int));
    dumpGraphviz(f, output, visited, total_values);
    free(visited);
    fprintf(f, "}\n");
    fclose(f);
}

int lossFunction(NN *nn) {
    BackpropValue **inputs = malloc(sizeof(BackpropValue*) * 3);
    for(int i = 0; i < 3; i++) {
        inputs[i] = malloc(sizeof(BackpropValue));
        createValue((float)i, inputs[i]); // Initialize with some values
    }

    BackpropValue **outputs = malloc(sizeof(BackpropValue*) * 2);
    callNN(nn, inputs, outputs);
    

    // calculate loss
    BackpropValue *act = NULL;
    BackpropValue *oldAct = NULL;

    for(int i = 0;i < 2; i++) {
        if(act != NULL) {
            oldAct = act;
        } else {
            oldAct = malloc(sizeof(BackpropValue));
            createValue(0.0f, oldAct);
        }

        act = malloc(sizeof(BackpropValue));
        createValue(0.0f, act); // Initialize activation value

        BackpropValue *target = malloc(sizeof(BackpropValue));
        createValue(((float)i)/3.0f, target); // Initialize target values

        BackpropValue *local_loss = malloc(sizeof(BackpropValue));
        createValue(0.0f, local_loss); // Initialize local loss value
        subtractValues(outputs[i], target, local_loss); // Calculate loss

        BackpropValue *local_loss_final = malloc(sizeof(BackpropValue));
        createValue(0.0f, local_loss_final); // Initialize final loss value
        multiplyValues(local_loss, local_loss, local_loss_final); // Square the loss

        addValues(oldAct, local_loss_final, act); // Accumulate the loss
    }

    BackpropValue *loss = malloc(sizeof(BackpropValue));
    createValue(0.0f, loss); // Initialize loss value
    
    BackpropValue *loss_normalizer = malloc(sizeof(BackpropValue));
    createValue(1.0f / 3.0f, loss_normalizer); // Normalizer for the loss
    multiplyValues(act, loss_normalizer, loss); // Normalize the loss
    printf("Loss: %f\n", loss->value);
    // Backward pass
    resetGrad(loss);
    backwardValue(loss);
    return 0;
}

int main() {
    parameters = malloc(sizeof(BackpropValue*) * 1);

    /*
    // EXAMPLE USAGE FOR BACKPROP VALUE
    BackpropValue *bv1 = malloc(sizeof(BackpropValue));
    BackpropValue *bv2 = malloc(sizeof(BackpropValue));
    BackpropValue *result = malloc(sizeof(BackpropValue));
    createValue(3.0f, bv1);
    createValue(4.0f, bv2);
    createValue(0.0f, result); // Initialize result value
    addValues(bv1, bv2, result);
    writeDotFile(result, "result.dot", 3);
    */

    /*
    // EXAMPLE USAGE FOR NEURON
    Neuron *neuron = malloc(sizeof(Neuron));
    createNeuron(5, neuron);
    BackpropValue **inputs = malloc(sizeof(BackpropValue*) * 5);
    for(int i = 0; i < 5; i++) {
        inputs[i] = malloc(sizeof(BackpropValue));
        createValue((float)i, inputs[i]); // Initialize with some values
    }
    BackpropValue *output = malloc(sizeof(BackpropValue));
    createValue(0.0f, output); // Initialize output value
    callNeuron(neuron, inputs, output);
    writeDotFile(output, "result.dot", 1024);
    */

    /*
    // EXAMPLE USAGE FOR LAYER CALLING
    Layer *layer = malloc(sizeof(Layer));
    createLayer(3, 2, layer);

    BackpropValue **inputs = malloc(sizeof(BackpropValue*) * 3);
    for(int i = 0; i < 3; i++) {
        inputs[i] = malloc(sizeof(BackpropValue));
        createValue((float)i, inputs[i]); // Initialize with some values
    }

    BackpropValue **outputs = malloc(sizeof(BackpropValue*) * 2);
    callLayer(layer, inputs, outputs);

    for(int i = 0; i < 2; i++) {
        displayValue(outputs[i]);
    }
    */

    NN *nn = malloc(sizeof(NN));
    createNN(3, 2, 6, nn, 20); // Create a neural network with 3 layers, input size 3, output size 2, and hidden neurons size 4
    printf("%d parameters\n", parameters_length);
    lossFunction(nn); // Calculate the loss and perform backpropagation

    for(int i = 0; i < parameters_length; i++) {
        // optimise the loss
        parameters[i]->value -= parameters[i]->grad * 0.01f; // Simple gradient descent step
    }
    lossFunction(nn); // Recalculate the loss after optimization
    

    return 0; 
}
