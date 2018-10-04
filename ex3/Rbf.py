import numpy as np


class Rbf(object):
    def __init__(self, input_shape, hidden_layers_shape, sigma=1.0):
        self.input_shape = input_shape
        self.hidden_shape = hidden_layers_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def gaussian_function(self, center, data):
        return np.exp(-self.sigma*np.linalg.norm(center-data)**2)

    def calculate_interpolation_matrix(self, input_data):
        interpolation_matrix = np.zeros((input_data.shape[0], self.hidden_shape))
        for data_point_arg, data_point in enumerate(input_data):
            for center_arg, center in enumerate(self.centers):
                interpolation_matrix[data_point_arg, center_arg] =\
                    self.gaussian_function(center, data_point)
        return interpolation_matrix

    def fit_and_train(self, input_data, labels):
        random_args = np.random.permutation(input_data.shape[0]).tolist()
        self.centers = [input_data[i] for i in random_args][:self.hidden_shape]
        interpolation_matrix = self.calculate_interpolation_matrix(input_data)
        self.weights = np.dot(np.linalg.pinv(interpolation_matrix), labels)

    def predict(self, input_data):
        # compute interpolation matrix
        interpolation_matrix = self.calculate_interpolation_matrix(input_data)
        # compute predictions
        predictions = np.dot(interpolation_matrix, self.weights)
        return predictions
