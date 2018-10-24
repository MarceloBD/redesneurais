import tensorflow as tf


class CNN():

    def __init__(self, num_classes, num_layers=3, filter_size=128,
                 kernel_size=(2, 2), pool_size=(2, 2)):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.model = None

    def create_network(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(self.filter_size,
                  self.kernel_size, input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(self.pool_size))
        model.add(tf.keras.layers.Dropout(0.1))
        for layer in range(1, self.num_layers):
            model.add(tf.keras.layers.Conv2D(layer*self.filter_size, self.kernel_size))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(self.pool_size))
            model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.num_classes,
                  activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
        return model

    def train_model(self, input_data, labels, num_epochs, batch_size):
        self.model.fit(input_data, labels,
                       epochs=num_epochs,
                       batch_size=batch_size)

    def validate(self, test_input, test_labels, steps):
        loss, accuracy = self.model.evaluate(test_input, test_labels,
                                             steps=steps)
        print("Loss:", loss, ",Accuracy:", accuracy)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

