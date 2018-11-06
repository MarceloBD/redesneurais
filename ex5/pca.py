import numpy as np 

class Pca():

	def __init__(self):
		return

	def cov_matrix(self, x, y, z):
		
		c = [[np.cov(x, x)[0][1], np.cov(x,y)[0][1], np.cov(x,z)[0][1]],
			[np.cov(y, x)[0][1], np.cov(y,y)[0][1], np.cov(y,z)[0][1]],
			[np.cov(z, x)[0][1], np.cov(z,y)[0][1], np.cov(z,z)[0][1]]]
		return c