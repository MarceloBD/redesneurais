from pca import Pca
import numpy as np
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_file():
    iris = load_iris()
    return iris.data, iris.target


colors = {0: 'ro',
          1: 'bo',
          2: 'go'}

if __name__ == '__main__':
    data, target = read_file()
    data = preprocessing.scale(data)
    #print(data, target)

    pca = Pca()
    cov = pca.cov_matrix(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    values, vectors = pca.eigen_values_vectors(cov)

#   print(values)
#   print(vectors)
    values, vectors = pca.sort_eigen(values, vectors)
#   print(values, vectors)

    vectors = pca.eigen_strip_vectors(values, vectors, 0.90)
#   print(vectors)
    values = values[:len(vectors[0])]

    #print(values)
    #print(data)
    #print(vectors)

    result = np.matrix.transpose(pca.pca_result(data,
                                 vectors)).reshape(len(data), len(data[0])-2)
    result[:, 1] = -result[:, 1]

    pca_result = decomposition.PCA(n_components=3)
    pca_result.fit(data)
    points = pca_result.transform(data)
    print(points)
    print(target)
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
        ax.plot(x, y, z, colors[label], label=str(count))
        count += 1
        class_points = []
    plt.grid()
    plt.show()
