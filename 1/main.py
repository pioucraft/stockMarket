from math import tanh
import random

class Value:
    def __init__(self, value, parents=(), op=None, grad=0.0):
        self.value = value
        self.parents = parents
        self.op = op
        self.grad = grad

    def __repr__(self):
        return f"(value: {self.value},  op: {self.op}, grad: {self.grad}, parents: {self.parents})"

    def __add__(self, other):
        return Value(self.value + other.value, (self, other), op='+')

    def __mul__(self, other):
        return Value(self.value * other.value, (self, other), op='*')

    def __pow__(self, other):
        return Value(self.value ** other.value, (self, other), op='**')
    
    def tanh(self):
        return Value(tanh(self.value), (self,), op='tanh') 

    def relu(self):
        return Value(max(0, self.value), (self,), op='relu')

    def backward(self, init=False):
        if init:
            self.grad = 1.0

        if self.op == "+":
            for parent in self.parents:
                parent.grad += self.grad
            for parent in self.parents:
                parent.backward()
        elif self.op == "*":
            self.parents[0].grad += self.grad * self.parents[1].value
            self.parents[1].grad += self.grad * self.parents[0].value
            for parent in self.parents:
                parent.backward()
        elif self.op == "**":
            self.parents[0].grad += self.grad * (self.parents[1].value * self.parents[0].value ** (self.parents[1].value - 1)) 
            self.parents[0].backward()
        elif self.op == 'tanh':
            self.parents[0].grad += self.grad * (1 - tanh(self.parents[0].value) ** 2)
            self.parents[0].backward()
        elif self.op == 'relu':
            if self.parents[0].value > 0:
                self.parents[0].grad += self.grad
            self.parents[0].backward()

    def reset_grad(self):
        self.grad = 0.0
        for parent in self.parents:
            parent.reset_grad()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        assert len(x) == len(self.w), "Input size must match weights size"
        z = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
        return z.relu()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

class MLP:
    def __init__(self, nin, nout, nhidden, nhin):
        self.layers = []
        self.layers.append(Layer(nin, nhin))
        for _ in range(nhidden):
            self.layers.append(Layer(nhin, nhin))
        self.layers.append(Layer(nhin, nout))
        
    deff __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
