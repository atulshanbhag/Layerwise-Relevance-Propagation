import tensorflow as tf
from utils import *
from model import MNIST_CNN

# Load data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

# Create a scope for the classifier
with tf.name_scope('MNIST_CNN'):
	model = MNIST_CNN()

	# Placeholders for data
	X = tf.placeholder(tf.float32, [None, 784], name='X')
	y = tf.placeholder(tf.float32, [None, 10], name='y')

	# Load model params and logits
	params, logits = model(X)

	# Add all params to Graph Collection
	tf.add_to_collection('DTD', X)
	for p in params:
		tf.add_to_collection('DTD', p)




