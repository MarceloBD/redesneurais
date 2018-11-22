from MLP import MLP
import data_handler as data
import numpy as np
from pca import Pca
from sklearn import preprocessing

if __name__ == '__main__':
    num_epochs = 1000
    # wine dataset
    # creating model
    mlp = MLP(3)
    # loading data
    inputs, labels = data.open_data('wine.arff', 3)
    inputs = np.array(inputs)
    labels = np.array(labels)

    # # without pca
    batch_size = 2
    # print(inputs[0])
    # leng = [len(inp) for inp in inputs]
    # print(max(leng), min(leng))
    # print(inputs.shape[1:])
    # mlp.create_network(inputs.shape[1:], 0.001)
    # mlp.train(inputs, labels, num_epochs, batch_size)
    # with pca
    pca = Pca()
    data = preprocessing.scale(inputs)
    cov = np.cov(data, rowvar=False)

    values, vectors = pca.eigen_values_vectors(cov)
    values, vectors = pca.sort_eigen(values, vectors)

    vectors = pca.eigen_strip_vectors(values, vectors, 0.98)

    print(vectors.shape)
    values = values[:len(vectors[0])]

    result = np.matrix.transpose(pca.pca_result(data,
                                 vectors)).reshape(len(data), 8)
    print(result.shape)
    points = result
    inputs = points
    mlp.create_network(inputs.shape[1:], 0.001)
    mlp.train(inputs, labels, num_epochs, batch_size)
