from mlp2 import Mlp 

if __name__ == '__main__':
	mlp = Mlp(3, 3, 2, 2)	   
	mlp.set_target([1,1,1])
	mlp.set_input([1,1,1])


	while(mlp.total_error() > 0.005):
		mlp.forward()
		mlp.backpropagation()
		mlp.update_weights(0.5)
		print(mlp.total_error())
