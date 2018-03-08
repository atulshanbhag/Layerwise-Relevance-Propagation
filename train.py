import tensorflow as tf
from utils import *
from model import MNIST_CNN

logdir = './logs/'
chkpt = './logs/model.ckpt'
n_epochs = 15
batch_size = 100

mnist = MNISTLoader()
mnist()
x_train, y_train = mnist.train
x_validation, y_validation = mnist.validation

with tf.name_scope('MNIST_CNN'):
	model = MNIST_CNN()

	X = tf.placeholder(tf.float32, [None, 784], name='X')
	y = tf.placeholder(tf.float32, [None, 10], name='y')

	params, logits = model(X)

	tf.add_to_collection('DTD', X)
	for p in params:
		tf.add_to_collection('DTD', p)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=model.params)

	preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(preds, tf.float32))

cost_summary = tf.summary.scalar(name='Cost', tensor=cost)
accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=accuracy)

summary = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

	train_batch = DataGenerator(x_train, y_train, batch_size)

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
		
		validation_accuracy = sess.run([accuracy, ], feed_dict={X: x_validation, y: y_validation})
		print(' Validation Accuracy {0:6.4f}'.format(validation_accuracy))

		saver.save(sess, chkpt)

