import random
from mlp import Mlp

target = [[1,1,0,0,1,1,0,0],
		  [1,1,1,1,1,1,1,1],
		  [0,0,1,1,0,0,1,1],
		  [1,0,1,0,1,0,1,0],
		  [0,0,0,0,0,0,0,0]]

learn_rate = 0.4
epochs = 0
mean_error = 100

if __name__ == '__main__':
	mpl = Mlp(8)

	mpl.set_target(target[0])
	mpl.update_all_neurons()
	while(mean_error > 0.1 and epochs < 5000):
		print (mean_error)
		mean_error = 0
		epochs += 1
		for t in range (5):
			mpl.set_target(target[t])
			mpl.update_all_neurons()
			while(mpl.total_error() > 0.2):
				for i in range(mpl.hidden_neurons*mpl.terminal_neurons):
				#	mpl.update_input_weights(i, mpl.hidden_error_derivative(i), learn_rate)
				#	mpl.update_hidden_weights(i, mpl.output_error_derivative(i), learn_rate)
					mpl.hidden_error_derivative(i)
					mpl.output_error_derivative(i)
				mpl.update_weights(learn_rate)
				mpl.update_all_neurons()
		for i in range(5):
			mpl.set_target(target[i])
			mpl.update_all_neurons()			
			mean_error += mpl.total_error()
		mean_error = mean_error/5

	mpl.print_outputs()
	print ("epochs utilized " + str(epochs))

	mpl.set_target(target[0])
	mpl.update_all_neurons()
	mpl.print_outputs()