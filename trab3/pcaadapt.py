import numpy as np 
import random

class PcaAdapt():

	def __init__(self, input_len):
		self.input_len = input_len
		self.outputs = []
		self.weights =  np.random.rand(input_len, input_len)	
		self.lateral_weights =  np.random.rand(input_len, input_len)	
		return

	def train(self, inputs):
		for inp in inputs:
			self.calculate_outputs(inp)
		return

	def calculate_outputs(self, inp):
		self.outputs = np.zeros(input_len)
		for j_out in range(self.input_len):
			for i_in in range(self.input_len): 
				self.outputs[j_out] +=  self.weights[i_in][j_out]*inp[i_in]
			for l in range(self.input_len):
				if(l < j):
					self.outputs[j_out] += self.lateral_weights[i_in][j_out]*self.outputs[j_out]
				else:
					break 
		return


 