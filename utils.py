import gzip
import pickle
import numpy as np

DATA_PATH = './mnist.pkl.gz'

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
		return self.X[start: end], self.y[start: end]	

class MNISTLoader:

	def __init__(self, loc=DATA_PATH):
		self.loc = loc

	def __call__(self):
		try:
			with gzip.open(DATA_PATH, 'rb') as f:
				data = pickle.load(f, encoding='bytes')
		except FileNotFoundError:
			print('Dataset not found!')
			exit()

		train_set, validation_set, test_set = data

		self.x_train, self.y_train = train_set
		self.x_validation, self.y_validation = validation_set
		self.x_test, self.y_test = test_set

		I = np.eye(10)
		self.y_train = I[self.y_train]
		self.y_validation = I[self.y_validation]
		self.y_test = I[self.y_test]

	def get_samples(self, n_samples, digit):
		data = [self.train, self.validation, self.test][np.random.choice(np.arange(3))]
		samples_indices = np.random.choice(np.argwhere(np.argmax(data[1], axis=1) == digit).flatten(), size=n_samples)
		return data[0][samples_indices]

	@property
	def train(self):
		return self.x_train, self.y_train

	@property
	def validation(self):
		return self.x_validation, self.y_validation

	@property
	def test(self):
		return self.x_test, self.y_test


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

	print('x_train shape', train[0].shape)
	print('y_train shape', train[1].shape)

	print('x_validation shape', validation[0].shape)
	print('y_validation shape', validation[1].shape)
	
	print('x_test shape', test[0].shape)
	print('y_test shape', test[1].shape)	

	dl.get_samples(1, 0)

