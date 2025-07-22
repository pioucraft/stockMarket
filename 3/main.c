#include "MLP.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define TYPE double 
#define LEARNING_RATE 1e-3

#define NUM_LAYERS 3
#define TRAINING_CYCLES 100


// Read big-endian 4-byte integer
uint32_t read_uint32(FILE *fp) {
    uint8_t bytes[4];
    fread(bytes, 1, 4, fp);
    return (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3];
}

// Read MNIST images
unsigned char* read_mnist_images(const char *filename, int *num_images, int *rows, int *cols) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot open %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32(fp);
    if (magic != 2051) {
        printf("Invalid MNIST image file!\n");
        fclose(fp);
        return NULL;
    }

    *num_images = read_uint32(fp);
    *rows = read_uint32(fp);
    *cols = read_uint32(fp);

    size_t size = (*num_images) * (*rows) * (*cols);
    unsigned char *data = (unsigned char*)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    return data;
}

// Read MNIST labels
unsigned char* read_mnist_labels(const char *filename, int *num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot open %s\n", filename);
        return NULL;
    }

    uint32_t magic = read_uint32(fp);
    if (magic != 2049) {
        printf("Invalid MNIST label file!\n");
        fclose(fp);
        return NULL;
    }

    *num_labels = read_uint32(fp);
    unsigned char *labels = (unsigned char*)malloc(*num_labels);
    fread(labels, 1, *num_labels, fp);
    fclose(fp);
    return labels;
}


int main() {

    // create some inputs and outputs
    int num_images, rows, cols, num_labels;
    unsigned char *images = read_mnist_images("data/train-images.idx3-ubyte", &num_images, &rows, &cols);
    unsigned char *labels = read_mnist_labels("data/train-labels.idx1-ubyte", &num_labels);

    TYPE **inputs = malloc(num_images * sizeof(TYPE*));
    TYPE **outputs = malloc(num_images * sizeof(TYPE*));

    for(int i = 0; i < num_images; i++) {
        unsigned char *img_ptr = images + i * rows * cols;
        for(int j = 0; j < rows * cols; j++) {
            if (inputs[i] == NULL) {
                inputs[i] = malloc(rows * cols * sizeof(TYPE));
            }
            inputs[i][j] = (TYPE)img_ptr[j] / 255.0; // Normalize pixel values to [0, 1] 
        }
        if (outputs[i] == NULL) {
            outputs[i] = malloc(10 * sizeof(TYPE)); // 10 classes for MNIST
            for(int j = 0; j < 10; j++) {
                if(j == labels[i]) {
                    outputs[i][j] = 1.0; // One-hot encoding
                } else {
                    outputs[i][j] = -1.0;
                }
            }
        }
    }

    num_images = 3000; // Limit to 3000 images for training

    // print the first input and output
    /*
    printf("First input: \n");
    for(int j = 0; j < rows * cols; j++) {
        // print in an ascii readable format using rows and cols 
        if (j > 0 && j % cols == 0) {
            printf("\n");
        }
        printf("%c", (inputs[0][j] > 0.5) ? '#' : '.');
    }
    printf("\nFirst output: ");
    for(int j = 0; j < 10; j++) {
        printf("%f \n", outputs[0][j]);
    }
    */


    int nin = rows * cols; // Number of input features
    int nout = 10; // Number of output neurons
    int nlayers = 4; // Total number of layers
    int n_neurons = 128;

    srand(42); // Seed for reproducibility
    NN* nn = createNN(nin, nout, nlayers, n_neurons);

    for (int i = 0; i < TRAINING_CYCLES; i++) {
        printf("Training cycle %d\n", i + 1);

        // calculate the current loss
        TYPE total_loss = 0.0;
        for (int j = 0; j < 100; j++) {
            TYPE* output = callNN(nn, inputs[j]);
            for (int k = 0; k < nout; k++) {
                TYPE error = outputs[j][k] - output[k];
                total_loss += error * error; // Mean Squared Error
            }
        }
        printf("Cycle %d: Loss = %f\n", i, total_loss / 100);

        for (int j = 0; j < num_images; j += 32) {
            int batch_size = (j + 32 > num_images) ? num_images - j : 32;
            reset_grad(nn);
            calculate_grad(nn, &inputs[j], &outputs[j], batch_size);
            optimise_parameters(nn, LEARNING_RATE, batch_size);
        }
        printf("Cycle %d: Gradients calculated and parameters updated.\n", i);

        TYPE* output = callNN(nn, inputs[num_images + i]);
        for(int k = 0; k < rows * cols; k++) {
            if (k > 0 && k % cols == 0) {
                printf("\n");
            }
            printf("%c", (inputs[num_images + i][k] > 0.5) ? '#' : '.');
        }
        printf("\nOutput: ");
        for(int k = 0; k < nout; k++) {
            printf("%d: %f ", k, output[k]);
        }
        printf("Actual: %d\n\n", labels[num_images + i]);

    }

    return 0;
}

