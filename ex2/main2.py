import numpy as np
from mult_layer_percepton import mlp

if __name__ == '__main__':
    dataset = np.identity(3)
    learn_rate = 0.1
    epochs = 10000
    model = mlp(1, 3, 3, 3)
    for epoch in range(epochs):
        error = []
        for data in dataset:
            error.append(model.backpropagation(data, data, learn_rate))
        print("epoch", epoch, "out of", epochs,
              "completed, loss:", sum(error)/len(error))
        np.random.shuffle(dataset)
    dataset = np.identity(3)
    print(dataset)
    for data in dataset:
        print(model.foward(data)[-1])
