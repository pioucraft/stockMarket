#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct BackpropValue {
    float value;
    struct BackpropValue *parents;
    int num_parents;
    char operation;
    float grad;
} BackpropValue;

int displayValueWithDepth(BackpropValue *bv, int depth) {
    printf("%p Value: %f, Parents: %p, NumParents: %d, Operation: %c, Grad : %f\n",
           (void *)bv, bv->value, (void *)bv->parents, bv->num_parents,
           bv->operation, bv->grad);
    if (depth > 0 && bv->parents != NULL) {
        for (int i = 0; i < bv->num_parents; i++) {
            displayValueWithDepth(&bv->parents[i], depth - 1);
        }
    }
    return 0;
}

int displayValue(BackpropValue *bv) { 
    return displayValueWithDepth(bv, 0); 
}

int addValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value + b->value;

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->num_parents = 2;
    result->operation = '+';
    return 0;
}

int subtractValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value - b->value;

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->num_parents = 2;
    result->operation = '-';
    return 0;
}

int multiplyValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value * b->value;

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->num_parents = 2;
    result->operation = '*';
    return 0;
}

int divideValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    if (b->value == 0) {
        fprintf(stderr, "Error: Division by zero\n");
        return -1;
    }
    result->value = a->value / b->value;

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->num_parents = 2;
    result->operation = '/';
    return 0;
}

int powerValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = pow(a->value, b->value);

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->num_parents = 2;
    result->operation = '^';
    return 0;
}

int reluValue(BackpropValue *a, BackpropValue *result) {
    result->value = a->value > 0 ? a->value : 0;

    result->parents = malloc(sizeof(BackpropValue));
    result->parents[0] = *a;
    result->num_parents = 1;
    result->operation = 'R';
    return 0;
}

int backwardValue(BackpropValue *bv) {
    // start by doing topological sort.
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

    return displayValueWithDepth(&test, 2);
}
