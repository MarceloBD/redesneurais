from mult_layer_perceptron import mlp
import numpy as np

if __name__ == '__main__':
    h_sizes = [3, 3]
    ml = mlp(h_sizes, 8, 8)
    data = np.random.uniform(-1, 1, 8)
    print(ml.foward(data), data)
