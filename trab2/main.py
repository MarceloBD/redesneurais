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
    batch_size = 500
    model = CNN(num_classes)
    data = []
    labels = []
    for i in range(5):
        dictionary = unpickle('cifar-10-batches-py/data_batch_' + str(i+1))
        data.append(dictionary[b'data'])
        labels.append(dictionary[b'labels'])
    data = [x/255 for x in data]
    data = np.reshape(data, (-1, img_size, img_size, 3))
    print(len(data))
    shape = data.shape[1:]
    labels = np.reshape(labels, len(labels)*len(labels[0]))
    labels = [make_one_shot_array(label, num_classes) for label in labels]
    labels = np.array(labels)
    model.create_network(shape)
    model.train_model(data, labels, num_epochs, batch_size)
