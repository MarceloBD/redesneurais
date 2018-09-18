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

    def foward(self, inputs):
        outputs = [inputs.copy()]
        next_input = [neuron.recognize(inputs)
                      for neuron in self.hidden_neurons[0]]
        outputs.append(next_input.copy())
        for i in range(1, len(self.hidden_neurons)):
            next_input = [neuron.recognize(next_input)
                          for neuron in self.hidden_neurons[i]]
            outputs.append(next_input.copy())
        next_input = [neuron.recognize(inputs)
                      for neuron in self.output_layer]
        outputs.append(next_input.copy())
        return outputs

    def backpropagation(self, data, labels, learn_rate):
        # computing layers outputs
        outputs = self.foward(data)
        # computing errors
        errors = [((label - predict)**2)/2.0
                  for label, predict in zip(labels, outputs[-1])]
        error = sum(errors)/len(errors)
        # reestimating weights from output layer

        # iterating over neurons
        for i in range(len(self.output_layer)):
            # iterating over weights
            for j in range(len(self.output_layer[i].weights)):
                # array[-1] is the last position of the array, so outputs[-1]
                # is the outputs of the out layer
                delta = (labels[i]-outputs[-1][i])*outputs[-1][i]*(1-outputs[-1][i])
                self.output_layer[i].weights[j] += learn_rate*delta*outputs[-2][j]
            self.output_layer[i].bias += learn_rate*delta
        # reestimating weights from hidden layers

        # iterating over layers, from the last to the first
        for z in range(len(self.hidden_neurons) - 1, -1, -1):
            # iterating over neurons
            for i in range(len(self.hidden_neurons[z])):
                # iterating over weights
                for j in range(len(self.hidden_neurons[z][i].weights)):
                    delta = [(labels[i]-outputs[z][k])*outputs[z][k]*(1-outputs[z][k])*outputs[z-1][i]
                             for k in range(len(self.hidden_neurons[z]))]
                    sum_delta = 0
                    for k in range(len(delta)):
                        if(z > 0):
                            sum_delta += delta[k] *\
                                        self.hidden_neurons[z-1][k].weights[i]
                        else:
                            sum_delta += delta[k]*data[k]
                    self.hidden_neurons[z][i].weights[j] += learn_rate *\
                        sum_delta*outputs[z][i]*(1-outputs[z][i])*outputs[z][j]
                    self.hidden_neurons[z][i].bias += learn_rate *\
                        sum_delta*outputs[z][i]*(1-outputs[z][i])
        return error
