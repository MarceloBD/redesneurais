from CNN import CNN
import numpy as np
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import os

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
    os.environ['CUDA_VISIBLE_DEVICES'] = str( 0 )
    num_epochs = 1000
    num_classes = 10
    img_size = 32
    batch_size = 100
    model = CNN(num_classes)
    #model.load('model.h5')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    shape = x_train.shape[1:]
    model.create_network(shape)
    model.train_model(x_train, y_train, num_epochs, batch_size)
    model.validate(x_test, y_test, 1)
    model.save('model.h5')

