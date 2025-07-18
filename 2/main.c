#include <stdio.h>
#include <stdlib.h>

typedef struct BackpropValue {
    float value;
    struct BackpropValue *parents;
    char operation;
} BackpropValue;

int displayValue(BackpropValue *bv) {
    printf("Value : %f, Parents: %p, Operation: %c\n", bv->value, (void*)bv->parents, bv->operation);
    return 0;
}

int addValues(BackpropValue *a, BackpropValue *b, BackpropValue *result) {
    result->value = a->value + b->value;

    result->parents = malloc(sizeof(BackpropValue) * 2);
    result->parents[0] = *a;
    result->parents[1] = *b;
    result->operation = '+';
    return 0; 
}

int main() {
    BackpropValue test;

    BackpropValue a;
    a.value = 10.0f;
    a.parents = NULL;
    BackpropValue b;
    b.value = 32.0f;
    b.parents = NULL;

    addValues(&a, &b, &test);

    return displayValue(&test);
}

