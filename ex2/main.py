import random
from mlp import Mlp
import numpy as np
import sys as sys

MATRIX_LEN = 10
learn_rate = 0.5
MIN_MEAN_ERROR = 0.01
MIN_SET_ERROR = 0.01
MAX_EPOCHS = 1
epochs = 0
mean_error = sys.maxsize


eye = [0 for _ in range(MATRIX_LEN**2)]
for i in range(MATRIX_LEN):
	eye[i*MATRIX_LEN+i]=1

target = [eye]

TRAINSET_LEN = np.size(target, 0)


if __name__ == '__main__':
	mpl = Mlp(MATRIX_LEN**2)
	mpl.set_target(target[0])
	mpl.update_all_neurons()
	number_weights_per_layer = mpl.hidden_neurons*mpl.terminal_neurons
	while(mean_error > MIN_MEAN_ERROR and epochs < MAX_EPOCHS):
		epochs += 1
		for t in range (TRAINSET_LEN):
			mpl.set_target(target[t])
			mpl.update_all_neurons()
			while(mpl.total_error() > MIN_SET_ERROR):
				mpl.calculate_output_delta()
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
				print(mpl.total_error())
		mean_error = 0
		for i in range(TRAINSET_LEN):
			mpl.set_target(target[i])
			mpl.update_all_neurons()			
			mean_error += mpl.total_error()
		mean_error = mean_error/TRAINSET_LEN
		print (mean_error)

	for i in range(TRAINSET_LEN):
		mpl.set_target(target[i])
		mpl.print_out_matrix(MATRIX_LEN)