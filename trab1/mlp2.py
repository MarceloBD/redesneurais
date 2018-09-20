import math 
import random 
import numpy as np

class Mlp():
    
    def __init__(self, input_neurons, output_neurons, hidden_neurons, hidden_layers):
        
        self.hidden_layers = hidden_layers 
        self.input_neurons = input_neurons + 1
        self.hidden_neurons = hidden_neurons + 1
        self.output_neurons = output_neurons + 1 

        self.number_weights_in_layer  = int((self.input_neurons )*(self.hidden_neurons-1))
        self.number_weights_hid_layer = int((self.hidden_neurons -1)*self.hidden_neurons)
        self.number_weights_out_layer = int(self.hidden_neurons*(self.output_neurons-1))

        self.weights = [random.uniform(0, 1) for _ in range(self.number_weights_in_layer)]
        for i in range(self.hidden_layers-1):
            new_hid_weight = [random.uniform(0, 1) for _ in range(self.number_weights_hid_layer)]
            self.weights = np.concatenate((self.weights, new_hid_weight),axis=0)
        output_weights = [random.uniform(0, 1) for _ in range(self.number_weights_out_layer)]
        self.weights = np.concatenate((self.weights, output_weights), axis=0)

        self.number_total_neurons = self.input_neurons  + self.output_neurons + self.hidden_layers * self.hidden_neurons
        self.output = [0 for _ in range(self.number_total_neurons)]

        self.first_output_neuron = self.input_neurons + self.hidden_neurons*self.hidden_layers 

        self.derivatives = [random.uniform(0, 1) for _ in range(len(self.weights))]
        self.last_update = [0 for _ in range(len(self.weights))]

        self.output[self.input_neurons-1] = 1
        for i in range(1,self.hidden_layers+1):
            self.output[i*hidden_neurons+self.input_neurons-1] = 1
        self.output[self.first_output_neuron+self.output_neurons-1] = 1

        self.target = [0 for _ in range(self.output_neurons-1)]
        self.output_delta  =  [0 for _ in range(self.output_neurons-1)]

    def set_target(self, target):
        for i in range(len(self.target)):
            self.target[i] = target[i]

    def set_input(self, input):
        for i in range(len(self.target)):
            self.output[i] = input[i]

    def total_error(self):
        error = 0
        for i in range(len(self.target)):
            error += 1/2.0 * (self.target[i] - self.output[self.first_output_neuron + i])**2 
        return error

    def activation_function(self, total_net_input):
        return 1.0/(1 + np.exp(-total_net_input))

    def forward(self):
        for i in range(self.hidden_neurons-1):
            self.output[i + self.input_neurons] = self.activation_function(self.total_net_input(i, 0))

        for i in range(1, self.hidden_layers):
            for j in range(self.hidden_neurons-1):
                self.output[i + self.input_neurons] = self.activation_function(self.total_net_input(j , i))

        for i in range(self.output_neurons - 1):
            self.output[self.first_output_neuron + i] = self.activation_function(self.total_net_input(i , self.hidden_layers))

    def total_net_input(self, neuron_number, layer_number):
        
        total_input = 0
        if(layer_number == 0):
            h_relative = neuron_number
            for i in range(self.input_neurons):
                total_input += self.output[i]*self.weights[i*(self.hidden_neurons-1)+h_relative]
        elif(layer_number < self.hidden_layers):
            first_neuron_layer = self.input_neurons + (layer_number-1)*self.hidden_neurons
            h_relative = neuron_number 
            first_weight_layer = self.number_weights_in_layer + (layer_number-1)*self.number_weights_hid_layer
            for i in range(self.hidden_neurons):
                total_input += self.output[i+first_neuron_layer]*self.weights[i*(self.hidden_neurons-1)+first_weight_layer+h_relative]
        else:
        	first_neuron_layer = self.input_neurons + (layer_number-1)*self.hidden_neurons
        	h_relative = neuron_number
        	first_weight_layer = self.number_weights_in_layer + (layer_number-1)*self.number_weights_hid_layer
        	for i in range(self.hidden_neurons):
        		total_input += self.output[i+first_neuron_layer]*self.weights[i*(self.output_neurons-1)+first_weight_layer+h_relative]
        return total_input

    def backpropagation_deltas(self):
        for i in range (self.output_neurons-1):
            o = i + self.first_output_neuron
            dEdOut = -(self.target[i]-self.output[o])
            dOutdNet = self.output[o]*(1-self.output[o])
            self.output_delta[i] = dEdOut*dOutdNet

    def backpropagation_output(self):
        for i in range(self.number_weights_out_layer):
            o_relative = int(i%(self.output_neurons-1)) 
            h_relative = int(i/(self.output_neurons-1))
            h = h_relative + self.input_neurons + (self.hidden_layers-1)*self.hidden_neurons
            o = o_relative + self.first_output_neuron
            dNetdW = self.output[h]
            first_output_weight = self.number_weights_in_layer+self.number_weights_hid_layer*(self.hidden_layers-1)
            self.derivatives[i+first_output_weight] = self.output_delta[o_relative]*dNetdW

    def backpropagation_hidden_first(self):
        for i in range(self.number_weights_in_layer):   
            i_relative = int(i/self.hidden_neurons)
            h_relative = int(i%self.hidden_neurons)
            h = h_relative + self.input_neurons             
            dEdOuth = 0
            for i in range(self.output_neurons-1):
                dNetdOut = self.weights[h_relative*(self.output_neurons-1)+i+self.number_weights_in_layer]
                dEdOuth += self.output_delta[i]*dNetdOut
                    
            dOutdNet = self.output[h]*(1-self.output[h])
            dNetdW = self.output[i_relative]
            self.derivatives[i] = dEdOuth*dOutdNet*dNetdW   

    def backpropagation_hidden_others(self, layer):
        for i in range(self.number_weights_hid_layer):  
            i_relative = int(i/self.hidden_neurons)
            h_relative = int(i%self.hidden_neurons)
            h = h_relative + self.input_neurons + self.hidden_neurons*layer            
            dEdOuth = 0
            for j in range(self.output_neurons-1):
                dNetdOut = self.weights[h_relative*self.output_neurons+j]
                dEdOuth += self.output_delta[j]*dNetdOut
                    
            dOutdNet = self.output[h]*(1-self.output[h])
            dNetdW = self.output[i_relative + self.input_neurons + self.hidden_neurons*(layer-1)]
            first_weight_layer = self.number_weights_in_layer + self.number_weights_hid_layer*(layer-1)
            self.derivatives[first_weight_layer+i] = dEdOuth*dOutdNet*dNetdW        
    
    def get_delta(self, out, hid, layer):
        if (layer > self.hidden_layers-1):
            return self.output_delta[out]
        else:
            layer += 1
            delta = 0
            for i in range():
                delta += get_delta(out, hid, layer)*self.output[i]*self.weights[i]
            return delta*self.output[hid]*(1-self.output[hid])
    

    def backpropagation(self):
        self.backpropagation_deltas()
        self.backpropagation_output()
        self.backpropagation_hidden_first()
        for i in range(1,self.hidden_layers-1):
            self.backpropagation_hidden_others(i)

    def update_weights(self, learn_rate, m):
        for i in range(len(self.weights)):
            self.weights[i] -= learn_rate*self.derivatives[i] + m*self.last_update[i]
            self.last_update[i] = learn_rate*self.derivatives[i] + m*self.last_update[i]

    def print_output(self):
        print("")
        for i in range(len(self.target)):
            print(self.output[self.first_output_neuron+i])

    def print_all_output(self):
    	print("")
    	for i in range(len(self.output)):
    		print(self.output[i])