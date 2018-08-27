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


class Mlp():
	
	def __init__(self, terminal_neurons):
		"""
		Constructs the class 
		"""
		self.terminal_neurons = terminal_neurons
		self.hidden_neurons = math.log(self.terminal_neurons, 2.0)
		number_weights_in_layer = self.hidden_neurons**self.terminal_neurons

		self.input_weights = [random.uniform(0, 1) for _ in range(number_weights_in_layer)]
		self.hidden_weights = [random.uniform(0, 1) for _ in range(number_weights_in_layer)]
		self.bias = [random.uniform(0, 1) for _ in range(3)]
		self.bias[0] = 0
		self.output = [0 for _ in range(2*self.terminal_neurons + self.hidden_neurons)]
		
	def set_target(self, target):
		"""
		Defines the expected output of the ouput neurons 
		"""
		self.target = target

	def total_error(self):
		"""
		Finds the total error in the output layer of the neural network
		"""
		error = 0
		first_output_neuron = self.terminal_neurons + self.hidden_neurons 
		for i in range(self.terminal_neurons):
			error += 1/2 * (target[i] - self.output[first_output_neuron + i])^2 
		return error

	def ouput_total_net_input(self, neuron_number):
		"""
		Finds the input of the output neuron
		"""
		total_input = 0

		for i in range(self.hidden_neurons):
			total_input += self.hidden_weights[i*self.terminal_neurons + neuron_number]*self.output[i + self.terminal_neurons]
		total_input += bias[2]
		return total_input 

	def hidden_total_net_input(self, neuron_number):
		"""
		Finds the input of the hidden neuron
		"""
		total_input = 0

		for i in range(self.terminal_neurons):
			total_input += self.input_weights[i*self.hidden_neurons + neuron_number]*self.output[i]
		total_input += bias[1]
		return total_input

	def output(self, total_net_input):
		"""
		Calculates the output of the neuron based in the logistic function
		(activation function)
		"""
		return 1/(1 + e^(-total_net_input))


	def output_error_derivative(self, respect_weight):
		"""
		Calculates error derivative for an output neuron with respect to a weight 
		"""
		h_relative = int(respect_weight/self.terminal_neurons)
		o_relative = respect_weight%self.terminal_neurons

		h = h_relative + self.terminal_neurons
		o = o_relative + self.terminal_neurons + self.hidden_neurons
		
		dEdOut = -(self.target[o_relative]-self.output[o])
		dOutdNet = self.output[o]*(1-self.output[o])
		dNetdW = self.output[h] 
		return dEdOut*dOutdNet*dNetdW

	def hidden_error_derivative(self, respect_weight):
		"""
		Calculates error derivative for an hidden neuron with respect to a weight 
		"""




