import numpy as np
from mult_layer_percepton import mlp

if __name__ == '__main__':
    data = [1, 0, 0]
    learn_rate = 0.5
    epochs = 100
    model = mlp(1, 3, 3, 3)
    model.backpropagation(data, data, epochs, learn_rate)
