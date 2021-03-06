from pca import Pca
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pcaadapt import PcaAdapt
import data_handler as data
from MLP import MLP

def read_file():
    iris = load_iris()
    return iris.data, iris.target

batch_size = 2
num_epochs = 1000
number_final_att = 13


if __name__ == '__main__':
	############################################## 2d
    inputs, labels = data.open_data('wine.arff', 3)
    inputs = np.array(inputs)
    labels = np.array(labels)

    pca = Pca()
    data = preprocessing.scale(inputs)

    pcaAdapt = PcaAdapt(13)
    pcaAdapt.train(data)

    result = np.matrix.transpose(pcaAdapt.pca_result(data)).reshape(len(data), number_final_att)    


    mlp = MLP(3)



    points = result
    inputs = points
    mlp.create_network(inputs.shape[1:], 0.001)
    mlp.train(inputs, labels, num_epochs, batch_size)

    '''  
    data, target = read_file()
    data = preprocessing.scale(data)

    pca = Pca()
    cov = pca.cov_matrix(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    values, vectors = pca.eigen_values_vectors(cov)
    values, vectors = pca.sort_eigen(values, vectors)

    vectors = pca.eigen_strip_vectors(values, vectors, 0.90)

    print(vectors)
    values = values[:len(vectors[0])]

    result = np.matrix.transpose(pca.pca_result(data,
                                 vectors)).reshape(len(data), len(data[0])-2)
    result[:, 1] = -result[:, 1]
    points = result
    count = 0
    class_points = []
    fig = plt.figure()
    for label in range(3):
        for i in range(len(target)):
            if(label == target[i]):
                class_points.append(points[i])
        class_points = np.array(class_points)
        x = class_points[:, 0]
        y = class_points[:, 1]
        plt.plot(x, y, colors[label], label=str(count))
        count += 1
        class_points = []
    plt.grid()
    plt.show()

    ############################################## 3d
    data, target = read_file()
    data = preprocessing.scale(data)

    pca = Pca()
    cov = pca.cov_matrix(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    values, vectors = pca.eigen_values_vectors(cov)
    values, vectors = pca.sort_eigen(values, vectors)

    vectors = pca.eigen_strip_vectors(values, vectors, 0.98)

    print(vectors)
    values = values[:len(vectors[0])]

    result = np.matrix.transpose(pca.pca_result(data,
                                 vectors)).reshape(len(data), len(data[0])-1)
    result[:, 1] = -result[:, 1]
    points = result
    count = 0
    class_points = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in range(3):
        for i in range(len(target)):
            if(label == target[i]):
                class_points.append(points[i])
        class_points = np.array(class_points)
        x = class_points[:, 0]
        y = class_points[:, 1]
        z = class_points[:, 2]
        plt.plot(x, y, z, colors[label], label=str(count))
        count += 1
        class_points = []
    plt.grid()
    plt.show()
    '''