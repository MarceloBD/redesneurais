from som import SOM
import data_handler as data

if __name__ == '__main__':
    inputs, labels = data.open_data('wine.arff', 3)
    print(labels)
    som = SOM(100, 100)
    som.train(inputs, 100000)
    som.plot_point_map(inputs, labels, ['Class 0', 'Class 1', 'Class 2'])
