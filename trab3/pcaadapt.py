import numpy as np 
import random

hebbian_rate = 1
alpha = 1
mu = 1

class PcaAdapt():

	def __init__(self, input_len):
		self.input_len = input_len
		self.outputs = []
		self.inputs = []
		self.weights =  np.random.rand(input_len, input_len)	
		self.lateral_weights =  np.random.rand(input_len, input_len)	
		return

	def train(self, inputs):
		for inp in inputs:
			self.calculate_outputs(inp)
		return

	def calculate_outputs(self, inp):
		self.outputs = np.zeros(self.input_len)
		self.inputs = inp
		for j_out in range(self.input_len):
			for i_in in range(self.input_len): 
				self.outputs[j_out] +=  self.weights[i_in][j_out]*inp[i_in]
			for l in range(self.input_len):
				if(l < j):
					self.outputs[j_out] += self.lateral_weights[i_in][j_out]*self.outputs[j_out]
				else:
					break 
		return

	def update_weights(self):
		for i in range(self.input_len):
			for j in range(self.input_len):
				self.weights[i][j] += hebbian_rate*self.outputs[j]*self.inputs[i] - alpha * self.outputs[j]**2 *self.weights[i][j]

	def update_lateral_weights(self):
		for i in range(self.input_len):
			for j in range(i, self.input_len):
				self.lateral_weights[i][j] += -mu * self.outputs[i] * self.outputs[j]

 