import tensorflow as tf


class MLP():
    def __init__(self, num_classes, num_layers=3,
                 layers_sizes=[128, 128, 256]):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers_sizes = layers_sizes
        if(len(layers_sizes) != num_layers):
            raise Exception("layers_sizes must have", num_layers, "length")

    def create_network(self, input_shape, learn_rate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.layers_sizes[0],
                                        input_shape=input_shape,
                                        activation='relu'))
        for layer in range(1, self.num_layers):
            model.add(tf.keras.layers.Dense(self.layers_sizes[layer],
                                            activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_classes,
                                        activation='softmax'))
        self.model = model
        optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, input_data, labels, num_epochs, batch_size, val_split=0.2):
        self.model.fit(input_data, labels, epochs=num_epochs,
                       batch_size=batch_size, validation_split=val_split)

    def evaluate(self, test_data, test_labels, steps=2):
        loss, accuracy = self.model.evaluate(test_data, test_labels,
                                             steps=steps)
        print("Model loss:", loss, ", Accuracy:", accuracy)
        return loss, accuracy
