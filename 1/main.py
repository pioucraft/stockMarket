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

    def __sub__(self, other):
        return Value(self.value - other.value, (self, other), op='-')

    def __mul__(self, other):
        return Value(self.value * other.value, (self, other), op='*')

    def __pow__(self, other):
        return Value(self.value ** other.value, (self, other), op='**')
    
    def tanh(self):
        return Value(tanh(self.value), (self,), op='tanh') 

    def relu(self):
        return Value(max(0, self.value), (self,), op='relu')

    def repr(self):
        return f"Value={self.value}"

    def backward(self, init=False):
        if init:
            self.grad = 1.0

        if self.op == "+":
            for parent in self.parents:
                parent.grad = self.grad
            for parent in self.parents:
                parent.backward()
        elif self.op == "-":
            self.parents[0].grad = self.grad
            self.parents[1].grad = self.grad
            for parent in self.parents:
                parent.backward()
        elif self.op == "*":
            self.parents[0].grad = self.grad * self.parents[1].value
            self.parents[1].grad = self.grad * self.parents[0].value
            for parent in self.parents:
                parent.backward()
        elif self.op == "**":
            self.parents[0].grad = self.grad * (self.parents[1].value * self.parents[0].value ** (self.parents[1].value - 1)) 
            self.parents[0].backward()
        elif self.op == 'tanh':
            self.parents[0].grad = self.grad * (1 - tanh(self.parents[0].value) ** 2)
            self.parents[0].backward()
        elif self.op == 'relu':
            if self.parents[0].value > 0:
                self.parents[0].grad = self.grad
            self.parents[0].backward()

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
        for i in range(len(x)):
            z = z + (self.w[i] * x[i])
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
            self.layers.append(Layer(nhin, nhin))
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
mlp = MLP(nin=3, nout=1, nhidden=2, nhin=4)
loss_history = []

for i in range(100):
    mlp.reset_grads()
    output = mlp(input_data)
    loss = Value(0.0)
    for o, e in zip(output, expected_output):
        loss = loss + (o - e) ** Value(2.0)
    loss.backward(init=True)
    for layer in mlp.layers:
        for neuron in layer.neurons:
            for j in range(len(neuron.w)):
                neuron.w[j].value -= 0.01 * neuron.w[j].grad
            neuron.b.value -= 0.01 * neuron.b.grad
    # visualize the parameters
    print(f"Parameters after iteration {i+1}:")
    print(f"Iteration {i+1}:")
    print("Output:", [o.value for o in output])
    print("Loss:", loss.value)
    loss_history.append(loss.value)

# Plotting the losss history with matplotlib
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
# save as png
plt.savefig('loss_history.png')

