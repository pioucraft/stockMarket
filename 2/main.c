#include <stdio.h>
#include <stdlib.h>

typedef struct BackpropValue {
    float value;
    struct BackpropValue **parents;
    int num_parents;
    char operation;
    float grad;
} BackpropValue;

int displayValueWithDepth(BackpropValue *bv, int depth) {
    printf("%p Value: %f, Parents: %p, NumParents: %d, Operation: %c, Grad : %f\n",
           (void *)bv, bv->value, bv->parents, bv->num_parents,
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

int reluValue(BackpropValue *a, BackpropValue *result) {
    result->value = a->value > 0 ? a->value : 0;

    result->parents = malloc(sizeof(BackpropValue*));
    result->parents[0] = a;
    result->num_parents = 1;
    result->operation = 'R';
    return 0;
}

int buildTopo(BackpropValue *bv, BackpropValue ***topo, int *index) {
    for(int i = 0; i < bv->num_parents; i++) {
        // topo = realloc(topo, sizeof(BackpropValue *) * (index + 1));
        // check if the parent has been visited
        int found = 0;
        for(int j = 0; j < *index; j++) {
            if ((*topo)[j] == (bv->parents[i])) {
                found = 1;
                break;
            }
        }
        if (found == 0) {
            *topo = realloc(*topo, sizeof(BackpropValue *) * (*index + 1));
            (*topo)[*index] = (bv->parents[i]);
            (*index)++;
            buildTopo(bv->parents[i], topo, index);
        }
    }
    return 0;
}

int _backwardValue(BackpropValue *bv) {
    printf("Current grad: %f\n", bv->grad);
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
    } else if(bv->operation == 'R') {
        if(bv->parents[0]->value > 0) {
            bv->parents[0]->grad += bv->grad;
        }
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

    // TODO : IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // IMPORTANT
    // YOU NEED TO BUILD THE TOPO IN BACKWARD ORDER, THEN REVERSE IT. TO HANDLE EDGE CASES (WHICH ARE NOT ""EDGE"" CASES WHEN BUUILDING MLP)
    buildTopo(bv, &topo, topo_length);

    bv->grad = 1.0f; // Set the gradient of the output value to 1.0
    _backwardValue(bv);
    for(int i = 0; i < *topo_length;i++) {
        _backwardValue(topo[i]);
    }

    free(topo);
    free(topo_length);

    displayValueWithDepth(bv, 2); // Display the values with depth 2
    
    return 0;
}


int main() {
    BackpropValue c;

    BackpropValue a;
    a.value = 10.0f;
    a.parents = NULL;
    a.num_parents = 0;
    a.operation = '_'; 
    a.grad = 0.0f;

    BackpropValue b;
    b.value = 32.0f;
    b.parents = NULL;
    b.num_parents = 0;
    b.operation = '_';
    b.grad = 0.0f;

    addValues(&a, &b, &c);

    BackpropValue test;
    multiplyValues(&a, &c, &test);

    resetGrad(&test);
    backwardValue(&test);

    return 0; 
}
