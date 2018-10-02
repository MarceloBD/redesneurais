from Mlp import Mlp
import data_handler as data

if __name__ == '__main__':
    num_classes = 3
    num_features = 7
    num_layers = 3
    layer_size = 128
    num_epochs = 2000
    learn_rate = 0.1
    batch_size = 21
    momentum = 0.8
    inputs, labels = data.open_data('seeds_dataset.txt', num_classes)
    test_inputs = inputs[int(0.8*len(inputs)):]
    test_labels = labels[int(0.8*len(labels)):]
    inputs = inputs[:int(0.8*len(inputs))]
    labels = labels[:int(0.8*len(labels))]
    neural_network = Mlp(num_layers, layer_size, num_features, num_classes)
    neural_network.train_and_evaluate(inputs, labels, test_inputs, test_labels,
                                      num_epochs, learn_rate, batch_size,
                                      momentum)
