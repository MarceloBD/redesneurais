"""
A library for multi layer perceptron with same number of neurons for each layer

The network is distributed as follows
	
	i0	h2	o4

	i1	h3	o5

	b0	b1	b2

	where b0 = 0 and where each neuron receives w_n weights

reference: Matt Mazur, A Step by Step Backpropagation Example 
				https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
"""
import random


class Mlp():
	
	def __init__(self, neurons_per_layer, num_layers):
		"""
		Constructs the class 
		"""
		self.weights = [random.uniform(0, 1) for _ in range(neurons_per_layer)]
		self.bias = [random.uniform(0, 1) for _ in range(num_layers)]
		self.output = [0 for _ in range(num_layers)]

	def neurons_per_layer(self):
		"""
		Returns the number of neurons each layer has, considering a homogeneous
		composition
		"""
		return len(self.weights)/number_of_layers()

	def number_of_layers(self):
		"""
		Returns the number of layers from perceptron 
		"""
		return len(self.bias)

	def total_error(self):
		"""
		Finds the total error in the output layer of the neural network
		"""
		error = 0
		first_output_neuron = neurons_per_layer()*(number_of_layers()-1)
		for i in range(neurons_per_layer()):
			error += 1/2 * (target[i] - self.output[first_output_neuron + i])^2 
		return error

	def total_net_input(self, neuron_number):
		"""
		Finds the input of the neuron
		"""
		total_input = 0
		for i in range(neurons_per_layer()):
			total_input += self.weights[i]*self.output[neuron_number-neurons_per_layer() + i]
		total_input += bias[int(neuron_number/neurons_per_layer())]
		return total_input 

	def output(self, total_net_input):
		"""
		Calculates the output of the neuron based in the logistic function
		(activation function)
		"""
		return 1/(1 + e^(-total_net_input))


	def output_error_derivative(respect_weight):
		"""
		Calculates error derivative for an output neuron with respect to a weight 
		"""
		position = respect_weight%(neurons_per_layer()^2)
		h_relative = int(position/neurons_per_layer())
		o_relative = position%neurons_per_layer()

		h = h_relative + (number_of_layers()*(number_of_layers()-2))
		o = o_relative + (number_of_layers()*(number_of_layers()-1))
		
		dEdOut = -(self.target[o_relative]-self.output[o])
		dOutdNet = self.output[o]*(1-self.output[o])
		dNetdW = self.output[h] 
		return dEdOut*dOutdNet*dNetdW

	def hidden_error_derivative(respect_weight):
		"""
		Calculates error derivative for an hidden neuron with respect to a weight 
		"""



	def set_target(target):
		self.target = target




