import numpy as np
from mult_layer_percepton import mlp

dataset = np.identity(3)
learn_rate = 0.1
epochs = 10000
mlp = mlp(1, 3, 10, 1)

if __name__ == '__main__':
   
    for epoch in range(epochs):
        error = []
        for data in dataset:
            error.append(mlp.backpropagation(data, data, learn_rate))
        print("epoch", epoch, "out of", epochs,
              "completed, loss:", sum(error)/len(error))
        np.random.shuffle(dataset)
    dataset = np.identity(3)
    print(dataset)
    for data in dataset:
        print(mlp.foward(data)[-1])
