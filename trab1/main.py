from mlp2 import Mlp 
import data_handler as data
import numpy as np

if __name__ == '__main__':
    # wine dataset
    # creating model
    mlp = Mlp(13, 3, 10, 1)
    # loading data
    inputs, labels = data.open_data('wine.arff', 3)
    print('class1', sum([label[0] for label in labels]))
    print('class2', sum([label[1] for label in labels]))
    print('class3', sum([label[2] for label in labels]))
    # separating training and tests
    test_inputs, test_labels = inputs[160:178], labels[160:178]
    inputs, labels = inputs[:160], labels[:160]
    # converting data into 1-d array
    for epoch in range(100):
        for i in range(len(inputs)):
            mlp.set_input(inputs[i])
            mlp.set_target(labels[i])
            mlp.forward()
            mlp.backpropagation()
            mlp.update_weights(0.1, 0.9)
        # inputs, labels = data.shuffle_data_and_labels(inputs, labels)
    # >>>>>>> 770f29e6958da69e40900b5b92f5685c31ad2aaa
    count = 0
    for input_data, label in zip(test_inputs, test_labels):
        mlp.set_input(input_data)
        mlp.set_target(label)
        out = mlp.print_output()
        print(out, label)
        if out == label:
            count += 1
    print(count, len(test_labels))
    print("Accuracy: ", count/len(test_labels))
    ###############################################################
    # default_features_1059_tracks
    dataset = data.open_data_tracks('default_features_1059_tracks.txt')
    dataset = np.reshape(dataset, -1)
    mlp = Mlp(70, 70, 10, 2)
    for epoch in range(1000):
        mlp.set_input(dataset)
        mlp.set_target(dataset)
        mlp.forward()
        mlp.backpropagation()
        mlp.update_weights(0.5, 0.8)
        print(mlp.total_error())
