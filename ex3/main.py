from Mlp import Mlp
from Rbf import Rbf
import data_handler as data
import numpy as np

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
    
    inputs = np.concatenate((inputs, inputs))
    labels = np.concatenate((labels, labels))
    accuracy = {}
    train_folds = (folds-1)/folds
    test_folds = 1/folds

    # cross-fold-validation
    for fold in range(folds):
        step = fold*int(test_folds*len(inputs)/2)
        train_set_len_step = int(train_folds*len(inputs)/2)+step
        test_set_len = int(test_folds*len(inputs)/2)

        train_inputs = inputs[step:train_set_len_step]
        train_labels = labels[step:train_set_len_step]

        test_inputs = inputs[train_set_len_step:train_set_len_step +
                             test_set_len]
        test_labels = labels[train_set_len_step:train_set_len_step +
                             test_set_len]
        neural_network = Mlp(num_layers, layer_size, num_features, num_classes)
        accuracy[fold] = neural_network.train_and_evaluate(train_inputs, train_labels, test_inputs, test_labels,
                                                           num_epochs, learn_rate, batch_size, momentum)
    print(sum(accuracy.values())/folds)
    

    inputs = np.concatenate((inputs, inputs))
    labels = np.concatenate((labels, labels))
    accuracy = {}
    train_folds = (folds-1)/folds
    test_folds = 1/folds

    # cross-fold-validation
    for fold in range(folds):
        step = fold*int(test_folds*len(inputs)/2)
        train_set_len_step = int(train_folds*len(inputs)/2)+step
        test_set_len = int(test_folds*len(inputs)/2)

        train_inputs = inputs[step:train_set_len_step]
        train_labels = labels[step:train_set_len_step]

        test_inputs = inputs[train_set_len_step:train_set_len_step +
                             test_set_len]
        test_labels = labels[train_set_len_step:train_set_len_step +
                             test_set_len]
        neural_network = Rbf(num_features, 128)
        for i in range(10):
            neural_network.fit(train_inputs, train_labels)
        predictions_temp = neural_network.predict(test_inputs)
        predictions = np.zeros((len(predictions_temp), num_classes))
        predictions = [list(map(int, x)) for x in predictions]
        for i in range(len(predictions)):
            predictions[i][np.argmax(predictions_temp[i])] = 1
        print(predictions)
        accur = sum([1 if (predict == label).all() else 0 for predict,label in zip(predictions, test_labels)])/len(test_labels)
        accuracy[fold] = accur
    print(sum(accuracy.values())/folds)
    
