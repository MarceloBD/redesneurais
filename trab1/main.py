from mlp2 import Mlp 
import data_handler as data
import numpy as np

if __name__ == '__main__':
    mlp = Mlp(13*13, 9, 2, 2)
    # loading data
    inputs, labels = data.open_data('wine.arff', 3)
    # separating training and tests
    test_inputs, test_labels = inputs[160:178], labels[160:178]
    inputs, labels = inputs[:160], labels[:160]
    # converting data into 1-d array
    inputs = np.reshape(inputs, (-1))
    labels = np.reshape(labels, (-1))
    mlp.set_input(inputs)
    mlp.set_target(labels)
    while(mlp.total_error() > 0.005):
        mlp.forward()
        mlp.backpropagation()
        mlp.update_weights(0.1)
        print(mlp.total_error())

    mlp.print_output()
