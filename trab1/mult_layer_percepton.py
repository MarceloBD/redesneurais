import numpy as np
from perceptron import Perceptron


class mlp():
    def __init__(self, n_hidden_layers, hidden_layer_size,
                 n_inputs, n_outputs):
        self.hidden_neurons = []
        self.hidden_neurons.append([Perceptron(n_inputs)
                                    for _ in range(hidden_layer_size)])
        for i in range(1, n_hidden_layers):
            self.hidden_neurons.append([Perceptron(hidden_layer_size)
                                        for _ in range(hidden_layer_size)])
        self.output_layer = [Perceptron(hidden_layer_size)
                             for _ in range(n_outputs)]

    def foward(self, x):
        next_input = [neuron.recognize(x) for neuron in self.hidden_neurons[0]]
        for i in range(1, len(self.hidden_neurons)):
            next_input = [neuron.recognize(next_input)
                          for neuron in self.hidden_neurons[i]]
        return [neuron.recognize(x) for neuron in self.output_layer]

    def backpropagation(self, data, labels, epochs):
        return
