from mlp2 import Mlp 

if __name__ == '__main__':
    mlp = Mlp(9, 9, 5, 4)    
    mlp.set_target([1,0,0,0,1,0,0,0,1])
    mlp.set_input([1,0,0,0,1,0,0,0,1])
    '''mlp.set_target([1, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 1])
    mlp.set_input([1,0,0,0,0,0,0,
                    0,1,0,0,0,0,0,
                    0,0,1,0,0,0,0,
                    0,0,0,1,0,0,0,
                    0,0,0,0,1,0,0,
                    0,0,0,0,0,1,0,
                    0,0,0,0,0,0,1])
    '''	
    first = 1

    while(mlp.total_error() > 0.0025 or first):
    	first = 0
    	mlp.forward()
    	mlp.backpropagation()
    	mlp.update_weights(0.4)
    	print(mlp.total_error())

    mlp.print_output()
   # mlp.print_all_output()