import tensorflow as tf 
import os 
feature_names = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove', 'variety']
record_defaults = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]


def decode_csv(line):
   parsed_line = tf.decode_csv(line, record_defaults)
   label =  parsed_line[-1]      
   del parsed_line[-1]                       
   features = tf.stack(parsed_line)  
   batch_to_return = features, label
   return batch_to_return

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(0).map(decode_csv))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(7)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

training_filenames = ["codebeautify.csv"]

#validation_filenames = ["dev_data1.csv"]

with tf.Session() as sess:     
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    while True:
        try:
        	features, labels = sess.run(next_element)
        	print("(train) features: ")
        	print(features)
        	print("(train) labels: ")
        	print(labels)  
        except tf.errors.OutOfRangeError:
        	print("Out of range error triggered (looped through training set 1 time)")
        	break
