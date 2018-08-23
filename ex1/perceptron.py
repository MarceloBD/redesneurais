import numpy as np
import random


class Perceptron():
    def __init__(self, num_inputs):
        self.weights = [random.uniform(0, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(0, 1)

    def generate_data(self, base_value):
        data = base_value.copy()
        noises_qtd = random.randint(1, len(data)/5)
        for i in range(noises_qtd):
            pos = int(random.uniform(0, len(data)))
            data[pos] = 1 if data[pos] == -1 else -1
        return data

    def write_text(self, data, name):
        with open(name, 'w') as file:
            for item in data:
                file.write("{}\n".format(item))

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
