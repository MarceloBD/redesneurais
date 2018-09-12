"""
A library for multi layer perceptron for encode and decode 

The network is distributed as follows
	
	i0	h0	o0

	i1	h1	o1
	
	..  ..  ..
	
	in  hln on

	b0	b1	b2

	where b0 = 0 and where input and output layers have n neurons with a log2n neuron
	hidden layer

reference: Matt Mazur, A Step by Step Backpropagation Example 
				https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
"""
import random
import math

class Mlp():
	
	def __init__(self, terminal_neurons):
		"""
		Constructs the class 
		"""
		self.terminal_neurons = terminal_neurons
		self.hidden_neurons = int(math.log(self.terminal_neurons, 2.0))
		number_weights_in_layer = int(self.hidden_neurons*self.terminal_neurons)
		self.hidden_weights = [random.uniform(0, 1) for _ in range(number_weights_in_layer)]
		self.output_weights = [random.uniform(0, 1) for _ in range(number_weights_in_layer)]
		self.bias = [random.uniform(0, 1) for _ in range(self.hidden_neurons+self.terminal_neurons)]
		self.output = [0 for _ in range(2*self.terminal_neurons + self.hidden_neurons)]
		self.hidden_derivatives = [0 for _ in range(number_weights_in_layer)]
		self.output_derivatives = [0 for _ in range(number_weights_in_layer)]
		self.output_delta = [0 for _ in range(self.terminal_neurons)]

		self.bias_out_derivatives = [0 for _ in range(self.terminal_neurons)]
		self.bias_hid_derivatives = [0 for _ in range(self.hidden_neurons)]

	def set_target(self, target):
		"""
		Defines the expected output of the ouput neurons and the input of the network 
		"""
		self.target = target
		for i in range(self.terminal_neurons):
			self.output[i] = target[i]

	def total_error(self):
		"""
		Finds the total error in the output layer of the neural network
		"""
		error = 0
		first_output_neuron = self.terminal_neurons + self.hidden_neurons 
		for i in range(self.terminal_neurons):
			error += 1/2.0 * (self.target[i] - self.output[first_output_neuron + i])**2 
		return error

	def update_all_neurons(self):
		"""
		Update the output for all neurons in the network 
		"""
		for i in range(self.hidden_neurons):
			self.output[self.terminal_neurons + i] = self.processes_input(self.hidden_total_net_input(i)) 
		for i in range (self.terminal_neurons):
			self.output[self.terminal_neurons + self.hidden_neurons + i] = self.processes_input(self.ouput_total_net_input(i))

	def ouput_total_net_input(self, neuron_number):
		"""
		Finds the input of the output neuron
		"""
		total_input = 0

		for i in range(self.hidden_neurons):
			total_input += self.output_weights[i*self.terminal_neurons + neuron_number]*self.output[i + self.terminal_neurons]
		
		total_input += self.bias[self.hidden_neurons + neuron_number]
		return total_input 

	def hidden_total_net_input(self, neuron_number):
		"""
		Finds the input of the hidden neuron
		"""
		total_input = 0

		for i in range(self.terminal_neurons):
			total_input += self.hidden_weights[i*self.hidden_neurons + neuron_number]*self.output[i]
		
		total_input += self.bias[neuron_number]
		return total_input

	def processes_input(self, total_net_input):
		"""
		Calculates the output of the neuron based in the logistic function
		(activation function)
		"""
		return 1/(1 + math.exp(-total_net_input))

	def calculate_output_delta(self):
		for i in range (self.terminal_neurons):
			o = i + self.terminal_neurons + self.hidden_neurons
			dEdOut = -(self.target[i]-self.output[o])
			dOutdNet = self.output[o]*(1-self.output[o])
			self.output_delta[i] = dEdOut*dOutdNet

	def output_error_derivative(self, respect_weight):
		"""
		Calculates error derivative for an output neuron with respect to a weight 
		"""
		h_relative = int(respect_weight/self.terminal_neurons)
		o_relative = int(respect_weight%self.terminal_neurons)

		h = h_relative + self.terminal_neurons
		o = o_relative + self.terminal_neurons + self.hidden_neurons
		
		dNetdW = self.output[h]

		self.output_derivatives[respect_weight] = self.output_delta[o_relative]*dNetdW

	def hidden_error_derivative(self, respect_weight):
		"""
		Calculates error derivative for an hidden neuron with respect to a weight 
		"""
		i_relative = int(respect_weight/self.hidden_neurons)
		h_relative = int(respect_weight%self.hidden_neurons)
		h = h_relative + self.terminal_neurons
		
		dEdOuth = 0
		first_output_neuron = self.terminal_neurons+self.hidden_neurons
		for i in range(self.terminal_neurons):
			dNetdOut = self.output_weights[h_relative*self.terminal_neurons+i]
			dEdOuth += self.output_delta[i]*dNetdOut
			
		dOutdNet = self.output[h]*(1-self.output[h])
		dNetdW = self.output[i_relative]
		self.hidden_derivatives[respect_weight] = dEdOuth*dOutdNet*dNetdW	

	def output_bias_derivative(self, respect_weight):
		o_relative = respect_weight
		o = o_relative + self.terminal_neurons + self.hidden_neurons
		
		dNetdW = 1 

		self.bias_out_derivatives[respect_weight] = self.output_delta[o_relative]*dNetdW	

	def update_bias_weights(self, learn_rate):
		for i in range(self.terminal_neurons):
			self.bias[self.hidden_neurons + i] -= learn_rate*self.bias_out_derivatives[i]
		for i in range(self.hidden_neurons):
			self.bias[i] -= learn_rate*self.bias_hid_derivatives[i]

	def hidden_bias_derivative(self, respect_weight):
		bias = respect_weight
		h_relative = bias
		h = h_relative + self.terminal_neurons

		dEdOuth = 0
		first_output_neuron = self.terminal_neurons+self.hidden_neurons
		for i in range(self.terminal_neurons):
			dNetdOut = self.output_weights[h_relative*self.terminal_neurons + i]
			dEdOuth += self.output_delta[i]*dNetdOut
			
		dOutdNet = self.output[h]*(1-self.output[h])
		dNetdW = 1 

		self.bias_hid_derivatives[respect_weight] =  dEdOuth*dOutdNet*dNetdW	

	def update_hidden_weights(self, weight_number, derivative, learn_rate):
		"""
		Decreases weitght of input neuron in network 
		"""
		self.hidden_weights[weight_number] -= learn_rate*derivative


	def update_output_weights(self, weight_number, derivative, learn_rate):
		"""
		Decreases weitght of input neuron in network 
		"""
		self.output_weights[weight_number] -= learn_rate*derivative


	def update_weights(self, learn_rate):
		for i in range(self.hidden_neurons*self.terminal_neurons):
			self.hidden_weights[i] -= learn_rate*self.hidden_derivatives[i]
			self.output_weights[i] -= learn_rate*self.output_derivatives[i]


	def print_out_matrix(self, len):
		for i in range(len):
			for j in range (len):
				print ("%.3f " % self.output[self.terminal_neurons+self.hidden_neurons+i*len+j],end='')
			print('')

	def print_outputs(self):
		"""
		Displays the output of the neural network
		"""
		for i in range(self.terminal_neurons):
			print (str(self.output[self.terminal_neurons+self.hidden_neurons+i])+' '+str(self.output[i]))
		print('')