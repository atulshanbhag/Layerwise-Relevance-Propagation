import tensorflow as tf
from utils import *
from model import MNIST_CNN

logdir = './logs/'
chkpt = './logs/model.ckpt'
n_epochs = 15
batch_size = 100

class Trainer:

	def __init__(self):
		self.dataloader = MNISTLoader()
		self.dataloader()
		
		self.x_train, self.y_train = self.dataloader.train
		self.x_validation, self.y_validation = self.dataloader.validation

		with tf.variable_scope('MNIST_CNN'):
			self.model = MNIST_CNN()

			self.X = tf.placeholder(tf.float32, [None, 784], name='X')
			self.y = tf.placeholder(tf.float32, [None, 10], name='y')

			self.activations, self.logits = self.model(self.X)

			tf.add_to_collection('DeepTaylorDecomposition', self.X)
			for act in self.activations:
				tf.add_to_collection('DeepTaylorDecomposition', act)

			self.l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in self.model.params if 'b' not in p.name]) * 0.0001
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)) + self.l2_loss
			self.optimizer = tf.train.AdamOptimizer().minimize(self.cost, var_list=self.model.params)

			self.preds = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
			self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))

		self.cost_summary = tf.summary.scalar(name='Cost', tensor=self.cost)
		self.accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=self.accuracy)

		self.summary = tf.summary.merge_all()

	def run(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

			self.train_batch = DataGenerator(self.x_train, self.y_train, batch_size)
			self.validation_batch = DataGenerator(self.x_validation, self.y_validation, batch_size)

			for epoch in range(n_epochs):
				self.train(sess, epoch)
				self.validate(sess)
				self.saver.save(sess, chkpt)

	def train(self, sess, epoch):
		n_batches = self.x_train.shape[0] // batch_size
		if self.x_train.shape[0] % batch_size != 0:
			n_batches += 1

		avg_cost = 0
		avg_accuracy = 0	
		for batch in range(n_batches):
			x_batch, y_batch = next(self.train_batch)
			_, batch_cost, batch_accuracy, summ = sess.run([self.optimizer, self.cost, self.accuracy, self.summary], 
																											feed_dict={self.X: x_batch, self.y: y_batch})
			avg_cost += batch_cost
			avg_accuracy += batch_accuracy
			self.file_writer.add_summary(summ, epoch * n_batches + batch)

			completion = batch / n_batches
			print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
			print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1), 
				print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
		print(end=' ')

	def validate(self, sess):
		n_batches = self.x_validation.shape[0] // batch_size
		if self.x_validation.shape[0] % batch_size != 0:
			n_batches += 1
		
		avg_accuracy = 0
		for batch in range(n_batches):
			x_batch, y_batch = next(self.validation_batch)
			avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: x_batch, self.y: y_batch})[0]

		avg_accuracy /= n_batches
		print('Validation Accuracy {0:6.4f}'.format(avg_accuracy))

if __name__ == '__main__':
	Trainer().run()
