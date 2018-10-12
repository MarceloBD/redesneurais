import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(data_train, labels_train), (data_test, labels_test) = mnist.load_data()

data_train = tf.keras.utils.normalize(data_train, axis=1)
data_test = tf.keras.utils.normalize(data_test, axis=1)

data_train = np.array(data_train).reshape(-1, 28, 28, 1)
data_test = np.array(data_test).reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(128, (3, 3), input_shape=data_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(128, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data_train, labels_train, epochs=100)

val_loss, val_acc = model.evaluate(data_test, labels_test)

print(val_loss, val_acc)
