from CNN import CNN
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def make_one_shot_array(value, num_classes):
    array = np.zeros(num_classes)
    array[value] = 1
    return array


if __name__ == '__main__':
    num_epochs = 1000
    num_classes = 10
    img_size = 32
    batch_size = 50
    model = CNN(num_classes)
    data = []
    labels = []
    for i in range(5):
        dictionary = unpickle('cifar-10-batches-py/data_batch_' + str(i+1))
        data.append(dictionary[b'data'])
        labels.append(dictionary[b'labels'])
    data = [x/255 for x in data]
    shape = (img_size, img_size, 3)
    model.create_network(shape)
    labels = np.reshape(labels, len(labels)*len(labels[0]))
    data = np.reshape(data, (len(data)*len(data[0]), img_size, img_size, 3))
    labels = [make_one_shot_array(x, num_classes) for x in labels]
    data = np.reshape(data, (-1, img_size, img_size, 3))
    labels = np.array(labels)
    model.train_model(data, labels, num_epochs, batch_size)
    dictionary = unpickle('cifar-10-batches-py/test_batch')
    test_data = dictionary[b'data']
    test_labels = dictionary[b'labels']
    test_data = np.reshape(test_data, (-1, img_size, img_size, 3))
    test_labels = [make_one_shot_array(x, num_classes) for x in test_labels]
    test_labels = np.array(test_labels)
    model.validate(test_data, test_labels, 10)

