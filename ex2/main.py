import random
from mlp import Mlp

target = [[1,1,0,0,1,1,0,0],
		  [1,1,0,1,0,1,1,1],
		  [0,0,1,1,0,0,1,1],
		  [1,0,1,0,1,0,1,0],
		  [0,1,0,1,0,1,0,0]]

learn_rate = 0.4
epochs = 0
mean_error = 100
min_error = 0.1
count_error = 100
last_error = 100
first = 0 

if __name__ == '__main__':
	mpl = Mlp(8)

	mpl.set_target(target[0])
	mpl.update_all_neurons()
	while(mean_error > 0.05 and epochs < 10000):
		if(first != 2):
			first += 1
			last_error = mean_error
		print (mean_error)
		epochs += 1
		for t in range (5):
			mpl.set_target(target[t])
			mpl.update_all_neurons()
			"""		
			if(count_error>1000 and (last_error - mean_error) < 0.1):
				min_error = min_error/1.2
				last_error = mean_error
				count_error = 0
				print ("---------------------------------divided")
			elif(count_error>1000):
				last_error = mean_error
				count_error = 0
			count_error += 1
			"""	  
			while(mpl.total_error() > min_error):
				for i in range(mpl.hidden_neurons*mpl.terminal_neurons):
				#	mpl.update_input_weights(i, mpl.hidden_error_derivative(i), learn_rate)
				#	mpl.update_hidden_weights(i, mpl.output_error_derivative(i), learn_rate)
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

	mpl.set_target([1,1,1,1,1,1,1,1])
	mpl.update_all_neurons()
	mpl.print_outputs()