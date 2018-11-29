import numpy as np
from som import SOM
import data_handler as data
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green']

if __name__ == '__main__':
    inputs, labels = data.open_data('wine.arff', 3)
    print(labels)
    som = SOM(100, 100)  # initialize the SOM
    som.fit(inputs, 100000)  # fit the SOM for 10000 epochs

    # now visualize the learned representation with the class labels
    som.plot_point_map(inputs, labels, ['Class 0', 'Class 1', 'Class 2'])

