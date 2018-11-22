import numpy as np 
import random

hebbian_rate = -0.001
alpha = 0.001
mu = 0.0007
number_final_att = 13
beta = 0.001

#hebbian_rate = 0.0009
#alpha = 0.002
#mu = 0.00001 
#number_final_att = 3
class PcaAdapt():

	def __init__(self, input_len):
		self.input_len = input_len
		self.outputs = []
		self.inputs = []
		self.weights =  np.random.rand(input_len, input_len)	
		self.lateral_weights =  np.random.rand(input_len, input_len)	
		self.old_delta = np.zeros((input_len, input_len))
		self.old_lateral_delta =  np.zeros((input_len, input_len))
		return

	def train(self, inputs):
	 #self.print_lateral_weights()
		for i in range(2):
			for inp in inputs:	
				self.calculate_outputs(inp)
				self.update_weights()
				self.update_lateral_weights()
			#self.print_lateral_weights()
			self.loss()
		self.save_eigen_vectors()
		return

	def save_eigen_vectors(self):
		self.eigen_vectors = np.empty((13,0), float)
		for i in range(number_final_att):
			self.eigen_vectors = np.hstack((self.eigen_vectors, self.weights[:,i].reshape(13,1)))

	def pca_result(self, data):
		return np.matmul(np.matrix.transpose(self.eigen_vectors), np.matrix.transpose(data))

	def print_lateral_weights(self):
		print('lateral')
		print(self.lateral_weights)
		#print('weights')
		#print(self.weights)

	def loss(self):
	#	print('loss: ', np.var(self.outputs)/np.linalg.norm(self.weights))

		lateral = 0
		for i in range(self.input_len):
			for j in range(i , self.input_len):
				lateral += np.abs(self.lateral_weights[i][j])
		print('e: ', lateral )

	def calculate_outputs(self, inp):
		self.outputs = np.zeros(self.input_len)
		self.inputs = inp
		for j_out in range(self.input_len):
			for i_in in range(self.input_len): 
				self.outputs[j_out] +=  self.weights[i_in][j_out]*inp[i_in]
			for l in range(self.input_len, j_out):
				self.outputs[j_out] += self.lateral_weights[l][j_out]*self.outputs[l]
		return

	def update_weights(self):
		for i in range(self.input_len):
			for j in range(self.input_len):
				delta = hebbian_rate*self.outputs[j]*self.inputs[i] - alpha * (self.outputs[j]**2) *self.weights[i][j]
				self.weights[i][j] += delta + beta*self.old_delta[i][j]
				self.old_delta[i][j] = delta
		return 

	def update_lateral_weights(self):
		for i in range(self.input_len):
			for j in range(i, self.input_len):
				lateral_delta = -mu * self.outputs[i] * self.outputs[j]
				self.lateral_weights[i][j] += lateral_delta + beta*self.old_lateral_delta[i][j]
				self.old_lateral_delta[i][j] = lateral_delta
		return
 