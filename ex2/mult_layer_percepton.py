import numpy as np
from perceptron import Perceptron


class mlp():
    def __init__(self, n_hidden_layers, hidden_layer_size,
                 n_inputs, n_outputs):
        self.hidden_neurons = [[Perceptron(n_inputs)
                               for _ in range(hidden_layer_size)].copy()]
        for i in range(1, n_hidden_layers):
            self.hidden_neurons += [[Perceptron(hidden_layer_size)
                                    for _ in range(hidden_layer_size)]]
        self.output_layer = [Perceptron(hidden_layer_size)
                             for _ in range(n_outputs)]

    def foward(self, x):
        y = [x.copy()]
        next_input = [neuron.recognize(x)
                      for neuron in self.hidden_neurons[0]]
        y.append(next_input.copy())
        for i in range(1, len(self.hidden_neurons)):
            next_input = [neuron.recognize(next_input)
                          for neuron in self.hidden_neurons[i]]
            y.append(next_input.copy())
        next_input = [neuron.recognize(x)
                      for neuron in self.output_layer]
        y.append(next_input.copy())
        return y

    def backpropagation(self, data, labels, epochs, learn_rate):
        for epoch in range(epochs):
            y = self.foward(data)
            # computing errors
            errors = [((label - predict)**2)/2
                      for label, predict in zip(labels, y[-1])]
            error = sum(errors)/len(errors)
            # reestimating weights from output layer
            for i in range(len(self.output_layer)):
                for j in range(len(self.output_layer[i].weights)):
                    delta = (labels[i]-y[-1][i])*y[-1][i]*(1-y[-1][i])
                    self.output_layer[i].weights[j] += learn_rate*delta*y[-2][j]
                self.output_layer[i].bias += learn_rate*delta
            # reestimating weights from hidden layers
            for z in range(len(self.hidden_neurons) - 1, -1, -1):
                for i in range(len(self.hidden_neurons[z])):
                    for j in range(len(self.hidden_neurons[z][i].weights)):
                        delta = [(labels[i]-y[z][k])*y[z][k]*(1-y[z][k])*y[z+1][k]
                                 for k in range(len(self.hidden_neurons[z]))]
                        sum_delta = 0
                        for k in range(len(delta)):
                            if(z > 0):
                                sum_delta += delta[k] *\
                                            self.hidden_neurons[z-1][k].weights[i]
                            else:
                                sum_delta += delta[k]*data[k]
                        self.hidden_neurons[z][i].weights[j] += learn_rate *\
                            sum_delta*y[z][i]*(1-y[z][i])*y[z+1][i]
                        self.hidden_neurons[z][i].bias += learn_rate *\
                            sum_delta*y[z][i]*(1-y[z][i])
            print("epoch", epoch, "out of", epochs,
                  "completed, loss:", error)
        print(self.foward(data)[-1])
