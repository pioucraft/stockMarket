from graphviz import Digraph
from math import tanh

def visualize_value(value, graph=None, seen=None):
    if graph is None:
        graph = Digraph(format='png')
        graph.attr(rankdir='LR')  # Left to right layout
    if seen is None:
        seen = set()

    # Unique id for each node based on Python's id()
    node_id = str(id(value))

    if node_id not in seen:
        seen.add(node_id)

        # Label includes value, grad, and op
        label = f"value={value.value:.4f}\ngrad={value.grad:.4f}"
        if value.op:
            label += f"\nop={value.op}"
        else:
            label += "\n(op=None)"

        graph.node(node_id, label=label, shape='ellipse', style='filled', fillcolor='lightyellow')

        for parent in value.parents:
            parent_id = str(id(parent))
            # Recursive call to add parent nodes
            visualize_value(parent, graph, seen)
            # Edge from parent to this node (reverse of computation flow)
            graph.edge(parent_id, node_id)

    return graph


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


# Example usage
a = (Value(2.0) + Value(3.0) * Value(4.0) + Value(5.0) ** Value(2.0)).relu() 
a.backward(init=True)
graph = visualize_value(a)
graph.render('value_graph', view=True)
