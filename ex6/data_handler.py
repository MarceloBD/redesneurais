import numpy as np
import csv
from sklearn.preprocessing import normalize as norm


def open_data(file_name, num_classes):
    data = []
    labels = []
    # oppening file
    with open(file_name, 'r') as file:
        # reading file
        data = list(csv.reader(file))
        # converting everything to float
        data = [list(map(float, x)) for x in data]
    # separating data and labels
    labels_temp = [int(label[0])-1 for label in data]
    # transforming labels into one_shot array
    labels = labels_temp
    # removing labels from data
    data = [np.delete(x, 0) for x in data]
    data, labels = shuffle_data_and_labels(data, labels)
    data = norm(data)
    labels = list(map(int, labels))
    return data, labels


def open_data_tracks(file_name):
    data = []
    # oppening file
    with open(file_name, 'r') as file:
        # reading file
        data = list(csv.reader(file))
        # converting everything to float
        data = [list(map(float, x)) for x in data]
    return norm(data)


def shuffle_data_and_labels(data, labels):
    aux = list(zip(data, labels))
    np.random.shuffle(aux)
    data[:], labels[:] = zip(*aux)
    return data, labels
