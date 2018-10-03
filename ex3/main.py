from Mlp import Mlp
import data_handler as data

if __name__ == '__main__':
    num_classes = 3
    num_features = 7
    num_layers = 3
    layer_size = 128
    num_epochs = 2000
    learn_rate = 0.01
    batch_size = 21
    momentum = 0.8
    folds = 10
    inputs, labels = data.open_data('seeds_dataset.txt', num_classes)
    
    inputs = inputs + inputs
    labels = labels + labels
    accuracy = {}
    train_folds = (folds-1)/folds
    test_folds = 1/folds

    #cross-fold-validation
    for fold in range(folds):
    	step = fold*int(test_folds*len(inputs)/2)
    	train_set_len = int(train_folds*len(inputs)/2)+step
    	test_set_len = int(test_folds*len(inputs)/2)

    	train_inputs = inputs[step:train_set_len]
    	train_labels = labels[step:train_set_len]

    	test_inputs = inputs[train_set_len:train_set_len+test_set_len]
    	test_labels = labels[train_set_len:train_set_len+test_set_len]
    	
    	neural_network = Mlp(num_layers, layer_size, num_features, num_classes)
    	accuracy[fold] = neural_network.train_and_evaluate(train_inputs, train_labels, test_inputs, test_labels,
                                      num_epochs, learn_rate, batch_size,
                                      momentum)
    print(sum(accuracy.values())/folds)
