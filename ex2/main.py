import random
from mlp import Mlp

target = [1,1,0,0,1,1,0,0]

learn_rate = 0.01
epochs = 0

if __name__ == '__main__':
	mpl = Mlp(8)
	mpl.set_target(target)
	mpl.update_all_neurons()


	while(mpl.total_error() > 0.01):
		epochs += 1
		for i in range(mpl.hidden_neurons*mpl.terminal_neurons):
			mpl.update_input_weights(i, mpl.hidden_error_derivative(i), learn_rate)
			mpl.update_hidden_weights(i, mpl.output_error_derivative(i), learn_rate)
		mpl.update_all_neurons()

	mpl.print_outputs()
	print ("epochs utilized " + str(epochs))