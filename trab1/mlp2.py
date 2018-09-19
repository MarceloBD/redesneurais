import math 
import random 
import numpy as np

class Mlp():
	
	def __init__(self, input_neurons, output_neurons, hidden_neurons, hidden_layers):
		
		self.hidden_layers = hidden_layers 
		self.input_neurons = input_neurons + 1
		self.hidden_neurons = hidden_neurons + 1
		self.output_neurons = output_neurons + 1 

		self.number_weights_in_layer  = int((self.input_neurons -1)*self.hidden_neurons)
		self.number_weights_hid_layer = int((self.hidden_neurons -1)*self.hidden_neurons)
		self.number_weights_out_layer = int((self.hidden_neurons -1)*self.output_neurons)

		self.weights = [random.uniform(0, 1) for _ in range(self.number_weights_in_layer)]
		for i in range(self.hidden_layers):
			new_hid_weight = [random.uniform(0, 1) for _ in range(self.number_weights_hid_layer)]
			self.weights = np.concatenate((self.weights, new_hid_weight),axis=0)
		output_weights = [random.uniform(0, 1) for _ in range(self.number_weights_out_layer)]
		self.weights = np.concatenate((self.weights, output_weights), axis=0)

		self.number_total_neurons = self.input_neurons  + self.output_neurons + self.hidden_layers * self.hidden_neurons
		self.output = [0 for _ in range(self.number_total_neurons)]

		self.first_output_neuron = self.input_neurons + self.hidden_neurons*self.hidden_layers 

		self.derivatives = [random.uniform(0, 1) for _ in range(self.number_weights_in_layer+self.number_weights_out_layer+self.number_weights_hid_layer*self.hidden_layers)]

		self.output[self.input_neurons-1] = 0
		for i in range(1,self.hidden_layers):
			self.output[i*hidden_neurons+self.input_neurons-1] = 1
		self.output[self.first_output_neuron+self.output_neurons-1] = 1


	def set_target(self, target):
		self.target = target

	def set_input(self, input):
		for i in range(self.input_neurons):
			self.output[i] = input[i]

	def total_error(self):
		error = 0
		for i in range(self.output_neurons):
			error += 1/2.0 * (self.target[i] - self.output[self.first_output_neuron + i])**2 
		return error

	def activation_function(self, total_net_input):
		return 1.0/(1 + math.exp(-total_net_input))

	def forward(self):
		for i in range(self.hidden_neurons):
			self.output[i + self.input_neurons] = activation_function(total_net_input(i + self.input_neurons, 0))

		for i in range(self.hidden_layers - 1):
			for j in range(self.hidden_neurons):
				self.output[i + self.input_neurons] = activation_function(total_net_input(i + self.input_neurons, i + 1))

		for i in range(self.output_neurons):
			self.output[self.first_output_neuron + i] = activation_function(total_net_input(i + self.first_output_neuron, self.hidden_layers))

	def total_net_input(self, neuron_number, layer_number):
		total_input = 0
		if(layer_number == 0):
			for i in range(self.input_neurons):
				total_input += self.output[i]*self.weights[i]
		else:
			first_neuron_layer = self.input_neurons+(layer_number-1)*self.hidden_neurons
			for i in range(self.hidden_neurons):
				total_input += self.output[i+first_neuron_layer]*self.weights[i+first_neuron_layer]
		return total_input


	def backpropagation(self):
		for i in range (self.output_neurons):
			o = i+self.first_output_neuron
			dEdOut = -(self.target[i]-self.output[o])
			dOutdNet = self.output[o]*(1-self.output[o])
			self.output_delta[i] = dEdOut*dOutdNet

		for i in range(self.number_weights_out_layer):
			o_relative = int(i%(self.output_neurons-1)) 
			h_relative = int(i/(self.output_neurons-1))
			h = h_relative + self.input_neurons + (self.hidden_layers-1)*self.hidden_neurons
			o = o_relative + self.first_output_neuron
			dNetdW = self.output[h]
			self.output_derivatives[respect_weight] = self.output_delta[o_relative]*dNetdW

		for i in range(self.number_weights_in_layer):	
			i_relative = int(i/self.hidden_neurons)
			h_relative = int(i%self.hidden_neurons)
			h = h_relative + self.input_neurons
				
			dEdOuth = 0
			for i in range(self.output_neurons):
				dNetdOut = self.weights[h_relative*self.terminal_neurons+i]
				dEdOuth += self.output_delta[i]*dNetdOut
					
			dOutdNet = self.output[h]*(1-self.output[h])
			dNetdW = self.output[i_relative]
			self.hidden_derivatives[respect_weight] = dEdOuth*dOutdNet*dNetdW	





