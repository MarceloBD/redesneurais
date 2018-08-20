import numpy as np
from random import uniform as rand


class Perceptron():
    def __init__(self, num_inputs):
        self.weights = [rand(0, 1) for _ in range(num_inputs)]
        self.bias = rand(0, 1)

    def generate_data(self, base_value, noise_val):
        data = base_value
        noises_qtd = int(rand(0, int(len(base_value)/5)))
        for i in range(noises_qtd):
            data[int(rand(0, len(base_value)))] = noise_val
        return data

    def activation_function(self, x):
        return 1 if x > 0 else -1

    def recognize(self, data):
        out = self.activation_function(sum([a*b for a, b in
                                            zip(data, self.weights)])
                                       + self.bias)
        return out

    def train(self, data, labels, epochs, learn_rate):
        for i in range(epochs):
            for x, y in zip(data, labels):
                out = self.recognize(x)
                if(out != y):
                    self.weights = [w + learn_rate*(y-out)*c for w, c in
                                    zip(self.weights, x)]
                    self.bias = self.bias + learn_rate*(y-out)
