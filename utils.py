import gzip
import pickle
import numpy as np

DATA_PATH = './mnist.pkl.gz'

# Load MNIST encoded as One-Hot labels
def load_data():
	try:
		with gzip.open(DATA_PATH, 'rb') as f:
			data = pickle.load(f, encoding='bytes')
	except FileNotFoundError:
		print('Dataset not found!')
		exit()

	train_set, val_set, test_set = data
	x_train, y_train = train_set
	x_val, y_val = val_set
	x_test, y_test = test_set

	# One-Hot labelling
	I = np.eye(10)
	y_train = I[y_train]
	y_val = I[y_val]
	y_test = I[y_test]

	return (x_train, y_train), (x_val, y_val), (x_test, y_test)

if __name__ == '__main__':
	(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
	print('x_train shape', x_train.shape)
	print('y_train shape', y_train.shape)
	
	print('x_val shape', x_val.shape)
	print('y_val shape', y_val.shape)	

	print('x_test shape', x_test.shape)
	print('y_test shape', y_test.shape)	
