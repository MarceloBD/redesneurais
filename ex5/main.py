from pca import Pca
import numpy as np 

def read_file():
	read_data = open("iris.data").read().lower()
	read_data = read_data.replace('\n',',')
	data_set = np.array([data for data in read_data.split(',')])
	data_set = data_set[:len(data_set)-2]
	data_set = data_set.reshape(int(len(data_set)/5), 5)
	return data_set

if __name__ == '__main__':
	data_set = read_file()
	print(data_set)