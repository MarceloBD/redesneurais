from mult_layer_perceptron import mlp
import numpy as np

if __name__ == '__main__':
    h_sizes = [3, 3]
    ml = mlp(h_sizes, 8, 8)
    data = np.identity(10)
    print(ml.foward(data), data)
