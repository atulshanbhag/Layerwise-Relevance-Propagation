import gzip
import pickle
import numpy as np

DATA_PATH = './mnist.pkl.gz'

# Batch Loader for Data
class DataGenerator:

	def __init__(self, X, y, batch_size):
		assert(X.shape[0] == y.shape[0])
		self.X = X
		self.y = y
		self.batch_size = batch_size
		self.num_samples = X.shape[0]
		self.num_batches = X.shape[0] // self.batch_size
		if X.shape[0] % self.batch_size != 0:
			self.num_batches += 1
		self.batch_index = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.batch_index == self.num_batches:
			self.batch_index = 0
		start = self.batch_index * self.batch_size
		end = min(self.num_samples, start + self.batch_size)
		self.batch_index += 1
		return self.X[start:end], self.y[start:end]	

# Data Loader for MNIST 
class MNISTLoader:

	def __init__(self, loc=DATA_PATH):
		self.loc = loc

	# Load MNIST encoded as One-Hot labels	
	def __call__(self):
		try:
			with gzip.open(DATA_PATH, 'rb') as f:
				data = pickle.load(f, encoding='bytes')
		except FileNotFoundError:
			print('Dataset not found!')
			exit()

		train_set, validation_set, test_set = data

		# Split into train, validation and test
		self.xtrain, self.ytrain = train_set
		self.xvalidation, self.yvalidation = validation_set
		self.xtest, self.ytest = test_set

		# One-Hot labelling
		I = np.eye(10)
		self.ytrain = I[self.ytrain]
		self.yvalidation = I[self.yvalidation]
		self.ytest = I[self.ytest]

	# Helper functions
	@property
	def train(self):
		return self.xtrain, self.ytrain

	@property
	def validation(self):
		return self.xvalidation, self.yvalidation

	@property
	def test(self):
		return self.xtest, self.ytest


if __name__ == '__main__':
	dl = MNISTLoader()
	dl()

	train = dl.train
	validation = dl.validation
	test = dl.test

	dg = DataGenerator(train[0], train[1], 100)
	for i in range(5):
		x, y = next(dg)
		print(i, x.shape, y.shape)

	print('xtrain shape', train[0].shape)
	print('ytrain shape', train[1].shape)

	print('xvalidation shape', validation[0].shape)
	print('yvalidation shape', validation[1].shape)
	
	print('xtest shape', test[0].shape)
	print('ytest shape', test[1].shape)	

