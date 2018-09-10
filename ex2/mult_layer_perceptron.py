import numpy as np
from numpy.random import uniform as rand


class mlp():
    def __init__(self, hidden_layers_sizes,
                 n_inputs, n_outputs):
        '''
        This function creates the mlp model
        @param hidden_layers_size: list of sizes of each hidden layer
        @param n_inputs: number of inputs
        @param n_outputs: number of outputs
        '''
        # first layer
        self.weights = [rand(-1, 1, (n_inputs, hidden_layers_sizes[0]))]
        self.biases = [rand(-1, 1, hidden_layers_sizes[0])]

        # remaining hidden layers
        for i in range(1, len(hidden_layers_sizes)):
            self.weights += [rand(-1, -1, (hidden_layers_sizes[i-1],
                                           hidden_layers_sizes[i]))]
            self.biases += [rand(-1, 1, hidden_layers_sizes[i])]

        # output layers
        self.weights += [rand(-1, 1, (hidden_layers_sizes[-1], n_outputs))]
        self.biases += [rand(-1, 1, n_outputs)]
        self.n_layers = len(self.biases)

    def foward(self, data):
        # v is the sum of the weights times biases
        v = []

        # Y is the result of the activation function applied to v
        Y = []
        for i in range(self.n_layers):
            aux = [sum(np.multiply(w[0], x)) + b
                   for w, x, b in zip(self.weights, data, self.biases)]
            v.append(aux)
            Y.append([self.activation_function(x) for x in aux])
        return v, Y

    def activation_function(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, y):
        return

    def recognize(self, data):
        return

    def backpropagation(self, data, labels, epochs):
        v, y = self.foward(data)
        self.backward(v, y)
        return
