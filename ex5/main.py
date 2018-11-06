from pca import Pca
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn import decomposition


def read_file():
	iris = load_iris()
	return iris.data, iris.target

if __name__ == '__main__':
	data, target = read_file()
	data = preprocessing.scale(data)
	print(data, target)

	pca = Pca()
	cov = pca.cov_matrix(data[:,0], data[:,1], data[:,2])
	
	values, vectors = pca.eigen_values_vectors(cov)
	pca.sort_eigen(values, vectors)





	print(cov)

	pca_result = decomposition.PCA(n_components=3)
	pca_result.fit(data)
	points = pca_result.transform(data)
	#print(points)
