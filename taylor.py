from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
import matplotlib.pyplot as plt

logdir = './logs/'
chkpt = './logs/model.ckpt'
digit = np.random.choice(10)

mnist = MNISTLoader()
mnist()
samples = mnist.get_samples(n_samples=1, digit=digit)

with tf.Session() as sess: 
  saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
  saver.restore(sess, tf.train.latest_checkpoint(logdir))

  weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MNIST_CNN')
  activations = tf.get_collection('DeepTaylorDecomposition')
  
  X = activations[0]

  act_weights = {}
  for act in activations[2:]:
    for wt in weights:
      name = act.name.split('/')[2]
      if name == wt.name.split('/')[2]:
        if name not in act_weights:
          act_weights[name] = []
        act_weights[name].append(wt)

  activations = activations[:0:-1]
  R = [activations[0], ] + [None, ] * 6

  w, b = act_weights['output']
  w_pos = tf.maximum(0.0, w)
  z = tf.nn.bias_add(tf.matmul(activations[1], w_pos), b) + 1e-10
  s = R[0] / z
  c = gen_nn_ops.bias_add_grad(tf.matmul(s, tf.transpose(w_pos)))
  R[1] = activations[1] * c
  
  w, b = act_weights['fc1']
  w_pos = tf.maximum(0.0, w)
  z = tf.nn.bias_add(tf.matmul(activations[2], w_pos), b) + 1e-10
  s = R[1] / z
  c = gen_nn_ops.bias_add_grad(tf.matmul(s, tf.transpose(w_pos)))
  R[2] = activations[2] * c

  R[3] = tf.reshape(R[2], [-1, 14, 14, 64])

  z = tf.nn.max_pool(activations[4], [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME') + 1e-10
  s = R[3] / z
  c = gen_nn_ops.max_pool_grad_v2(activations[4], z, s, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  R[4] = activations[4] * c

  w, b = act_weights['conv2']
  w_pos = tf.maximum(0.0, w)
  z = tf.nn.bias_add(tf.nn.conv2d(activations[5], w_pos, [1, 1, 1, 1], padding='SAME'), b) + 1e-10
  s = R[4] / z
  c = gen_nn_ops.bias_add_grad(tf.nn.conv2d_backprop_input(tf.shape(activations[5]), w_pos, s, [1, 1, 1, 1], padding='SAME'))
  R[5] = activations[5] * c

  w, b = act_weights['conv1']
  w_pos = tf.maximum(0.0, w)
  z = tf.nn.bias_add(tf.nn.conv2d(activations[6], w_pos, [1, 1, 1, 1], padding='SAME'), b) + 1e-10
  s = R[5] / z
  c = gen_nn_ops.bias_add_grad(tf.nn.conv2d_backprop_input(tf.shape(activations[6]), w_pos, s, [1, 1, 1, 1], padding='SAME'))
  R[6] = activations[6] * c

  heatmap = sess.run(R[6], feed_dict={X: samples})[0].reshape(28, 28)
  heatmap /= heatmap.max()
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.axis('off')
  ax.set_title(str(digit))
  ax.imshow(heatmap, cmap='Reds', interpolation='bilinear')
  fig.savefig('img.jpg')

  plt.show()

