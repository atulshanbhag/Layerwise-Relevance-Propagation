import tensorflow as tf
from utils import *
from model import MNIST_CNN

logdir = './logs/'
chkptdir = './logs/model/'
n_epochs = 15
batch_size = 100

# Load data
mnist = MNISTLoader()
mnist()
x_train, y_train = mnist.train

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
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
	
	# RSMProp optimizer with default parameters
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost, var_list=model.params)

	# Retrieve model predictions and calculate accuracy
	preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(preds, tf.float32))

# Create summaries for TensorBoard visualization
cost_summary = tf.summary.scalar(name='Cost', tensor=cost)
accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)

# Merge all summaries as one summary
summary = tf.summary.merge_all()

# Train the network
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Save logs and model checkpoints
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Batch Loader for Training set
train_batch = DataGenerator(x_train, y_train, batch_size)

# Training
print('Training...')
for epoch in range(n_epochs):
	n_batches = x_train.shape[0] // batch_size
	if x_train.shape[0] % batch_size != 0:
		n_batches += 1
	avg_cost = 0
	avg_accuracy = 0

	for batch in range(n_batches):
		 x_batch, y_batch = next(train_batch)
		 _, batch_cost, batch_accuracy, summ = sess.run([optimizer, cost, accuracy, summary], 
		 																								feed_dict={X: x_batch, y: y_batch})
		 avg_cost += batch_cost
		 avg_accuracy += batch_accuracy
		 file_writer.add_summary(summ, epoch * n_batches + batch)

		 completion = batch / n_batches
		 print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
		 print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1), 
						print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
	print()
	saver.save(sess, chkptdir)

print('Accuracy: {0;.4f}'.format(sess.run(accuracy, feed_dict={X: x_train, y: y_train})))
sess.close()    


