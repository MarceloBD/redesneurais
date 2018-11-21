import numpy as np 

class Pca():

	def __init__(self):
		return

	def cov_matrix(self, x, y, z, l):
		
		c = [[np.cov(x, x)[0][1], np.cov(x,y)[0][1], np.cov(x,z)[0][1], np.cov(x,l)[0][1]],
			[np.cov(y, x)[0][1], np.cov(y,y)[0][1], np.cov(y,z)[0][1], np.cov(y,l)[0][1]],
			[np.cov(z, x)[0][1], np.cov(z,y)[0][1], np.cov(z,z)[0][1], np.cov(z,l)[0][1]],
			[np.cov(l, x)[0][1], np.cov(l,y)[0][1], np.cov(l,z)[0][1], np.cov(l,l)[0][1]]]
		return c

	def eigen_values_vectors(self, cov):
		return np.linalg.eig(cov)

	def sort_eigen(self, values, vectors):
		ordered_index = np.argsort(-np.array(values))[:len(values)]
		values_ordered = [values[i] for i in ordered_index]
		vectors_ordered = np.empty((4,0), float)
		for i in ordered_index:
			vectors_ordered = np.hstack((vectors_ordered, vectors[:,i].reshape(4,1)))
		return [values_ordered, vectors_ordered]

	def eigen_strip_vectors(self, values, vectors, var_min):	
		var = 0
		vec = np.empty((4,0), float)
		for i in range(len(values)):
			vec = np.hstack((vec, vectors[:,i].reshape(4,1)))
			var += values[i]
			if(var/np.sum(values) > var_min):
				break; 
		return vec

	def pca_result(self, data, vectors):
		return np.matmul(np.matrix.transpose(vectors), np.matrix.transpose(data))