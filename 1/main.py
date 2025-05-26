from math import tanh

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
