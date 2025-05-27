from math import tanh, exp, log
import random
import numpy as np

class Value:
    def __init__(self, value, parents=(), op=None, grad=0.0):
        self.value = value
        self.parents = parents
        self.op = op
        self.grad = grad
        self._backward = lambda:None

    def __repr__(self):
        return f"(value: {self.value},  op: {self.op}, grad: {self.grad}, parents: {self.parents})"

    def __add__(self, other):
        out = Value(self.value + other.value, (self, other), op='+')
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __sub__(self, other):
        out = Value(self.value - other.value, (self, other), op='-')
        def backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        out = Value(self.value * other.value, (self, other), op='*')
        def backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = backward
        return out

    def __pow__(self, other):
        out = Value(self.value ** other.value, (self, other), op='**')
        def backward():
            self.grad += out.grad * (other.value * self.value ** (other.value - 1))
        out._backward = backward
        return out 

    def tanh(self):
        out = Value(tanh(self.value), (self,), op='tanh')
        def backward():
            self.grad += out.grad * (1 - tanh(self.value) ** 2)
        out._backward = backward
        return out

    def relu(self):
        out_val = log(1 + exp(self.value))
        out = Value(out_val, (self,), op='softplus')
        def backward():
            self.grad += out.grad * (1 / (1 + exp(-self.value)))
        out._backward = backward
        return out

    def repr(self):
        return f"Value={self.value}"

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def reset_grad(self):
        self.grad = 0.0
        for parent in self.parents:
            parent.reset_grad()


class Neuron:
    def __init__(self, nin, relu=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.relu = relu
    
    def __call__(self, x):
        assert len(x) == len(self.w), "Input size must match weights size"
        z = Value(0.0) 
        z = z + np.dot(self.w, x)
        z = z + self.b
        if self.relu:
            return z.relu()
        else:
            return z.tanh()


class Layer:
    def __init__(self, nin, nout, relu=True):
        self.neurons = [Neuron(nin, relu) for _ in range(nout)]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

class MLP:
    def __init__(self, nin, nout, nhidden, nhin):
        self.layers = []
        self.layers.append(Layer(nin, nhin))
        for _ in range(nhidden):
            self.layers.append(Layer(nhin, nhin, relu=False))
        self.layers.append(Layer(nhin, nout, relu=False))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reset_grads(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for w in neuron.w:
                    w.reset_grad()
                neuron.b.reset_grad()

input_data = [Value(1.0), Value(2.0), Value(3.0)]
expected_output = [Value(0.5)]
mlp = MLP(nin=3, nout=1, nhidden=10, nhin=10)
loss_history = []

for i in range(1000):
    mlp.reset_grads()
    output = mlp(input_data)
    loss = Value(0.0)
    for o, e in zip(output, expected_output):
        loss = loss + (o - e) ** Value(2.0)
    loss.backward()
    for layer in mlp.layers:
        for neuron in layer.neurons:
            for j in range(len(neuron.w)):
                neuron.w[j].value -= 0.01 * neuron.w[j].grad
            neuron.b.value -= 0.01 * neuron.b.grad
    # visualize the parameters
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.value}")
    loss_history.append(loss.value)

# Plotting the losss history with matplotlib
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
# save as png
plt.savefig('loss_history.png')

