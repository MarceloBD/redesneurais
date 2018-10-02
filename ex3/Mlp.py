import tensorflow as tf
import numpy as np


class Mlp():
    def __init__(num_layers, layer_size, num_inputs, num_outputs):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = tf.placeholder(tf.float32, [None, num_inputs])
        self.outputs = tf.placeholder('float')

    def neural_network(self, input_data):
        # initializing weights and biases
        weights = {}
        biases = {}
        # first layer
        weights[0] = tf.Variable(tf.random_normal([self.num_inputs,
                                                   self.layer_size]))
        biases[0] = tf.Variable(tf.random_normal([self.layer_size]))
        # remaining hidden layers
        for layer in range(1, self.num_layers):
            weights[layer] = tf.Variable(tf.random_normal([self.layer_size,
                                                           self.layer_size]))
            biases[layer] = tf.Variable(tf.random_normal([self.layer_size]))
        # output layer
        layer += 1
        weights[layer] = tf.Variable(tf.random_normal([self.layer_size,
                                                       n_classes]))
        biases[layer] = tf.Variable(tf.random_normal([n_classes]))

        layer_out = {-1: input_data}

        # compute output of each layer
        for layer in weights:
            # (inputs * weights) + bias
            layer_out[layer] = tf.add(tf.matmul(layer_out[layer - 1],
                                      weights[layer]), biases[layer])
            layer_out[layer] = tf.nn.sigmoid(layer_out[layer])
        # in the output layer we will aplly softmax
        layer_out[layer] = tf.add(tf.matmul(layer_out[layer - 1],
                                  weights[layer]), biases[layer])
        return layer_out[layer]

    def train_and_evaluate(self, inputs, labels, test_inputs, test_labels,
                           num_epochs, learn_rate, batch_size):
        prediction = self.neural_network(self.inputs)
        cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                           labels=self.outputs)
                )
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                epoch_loss = 0
                i = 0
                while i < len(inputs):
                    epoch_x = np.array(inputs[i:i+batch_size])
                    epoch_y = np.array(labels[i:i+batch_size])
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                    i += batch_size
                print('Epoch', epoch, 'completed out of',
                      num_epochs, 'loss:', epoch_loss)
                if(epoch % 200 == 0):
                    correct = tf.equal(tf.argmax(prediction, 1),
                                       tf.argmax(self.outputs, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    print('Accuracy:', accuracy.eval({x: np.array(test_inputs),
                                                      y: np.array(test_labels)
                                                      }))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: np.array(test_inputs),
                  y: np.array(test_labels)}))
