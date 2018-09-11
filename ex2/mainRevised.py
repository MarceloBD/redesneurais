import random
from mlp import Mlp
import numpy as np
import sys as sys

matrix = [0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0,\
		  0,0,0,0,0,0,0,0,0,0]


new_matrix = matrix.copy()
new_matrix[0] = 1
target = [new_matrix]		 

for i in range(1,10):
	new_matrix = matrix.copy()
	new_matrix[i*10+i] = 1
	target = np.vstack([target, new_matrix])

learn_rate = 0.5
MIN_MEAN_ERROR = 0.005
MIN_SET_ERROR = 0.01
MAX_EPOCHS = 1000
epochs = 0
mean_error = sys.maxsize

first = 0 

TRAINSET_LEN = np.size(target, 0)
MATRIX_LEN = np.size(target, 1)

if __name__ == '__main__':
	mpl = Mlp(MATRIX_LEN)
	mpl.set_target(target[0])
	mpl.update_all_neurons()
	number_weights_per_layer = mpl.hidden_neurons*mpl.terminal_neurons
	while(mean_error > MIN_MEAN_ERROR and epochs < MAX_EPOCHS):
		epochs += 1
		for t in range (TRAINSET_LEN):
			mpl.set_target(target[t])
			mpl.update_all_neurons()
			while(mpl.total_error() > MIN_SET_ERROR):
				for i in range(number_weights_per_layer):
					mpl.hidden_error_derivative(i)
					mpl.output_error_derivative(i)
				for i in range(mpl.terminal_neurons):
					mpl.output_bias_derivative(i)
				for i in range(mpl.hidden_neurons):
					mpl.hidden_bias_derivative(i)
				mpl.update_bias_weights(learn_rate)
				mpl.update_weights(learn_rate)
				mpl.update_all_neurons()
		
		mean_error = 0
		for i in range(TRAINSET_LEN):
			mpl.set_target(target[i])
			mpl.update_all_neurons()			
			mean_error += mpl.total_error()
		mean_error = mean_error/TRAINSET_LEN
		print (mean_error)

	print ("epochs utilized " + str(epochs))
	for i in range(TRAINSET_LEN):
		mpl.set_target(target[i])
		mpl.update_all_neurons()
		mpl.print_outputs()