import tensorflow as tf
from utils import *
from model import MNIST_CNN

LOGDIR = './logs/'
N_EPOCHS = 15
BATCH_SIZE = 100

# Load data
mnist = MNISTLoader()
mnist()
xtrain, ytrain = mnist.train
xvalidion, yvalidation = mnist.validation
xtest, ytest = mnist.test

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

	# Cross-Entropy Loss
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	
	# RSMProp optimizer with default parameters
	optimizer = tf.train.RMSPropOptimizer().minimize(cost, model.params)

	# Retrieve model predictions and calculate accuracy
	preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(preds, tf.float32))

# Create summaries for TensorBoard visualization
cost_summary = tf.summary.scalar(name='Cost', tensor=cost)
accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)

# Merge all summaries as one summary
summary = tf.summary.merge_all()

# Train the network
with tf.InteractiveSession() as sess:
	sess.run(tf.global_variables_initializer())

	# Save logs and model checkpoints
	saver = tf.train.Saver()
	file_writer = tf.summary.FileWriter(LOGDIR, tf.get_default_graph())

	for epoch in range(N_EPOCHS):
		n_batches = x_train.shape[0] // BATCH_SIZE
		avg_cost = 0
		avg_accuracy = 0

		for batch in range(n_batches):
			pass 


