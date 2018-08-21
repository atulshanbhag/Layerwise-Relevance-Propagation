from utils                    import MNISTLoader
from tensorflow.python.ops    import gen_nn_ops

import numpy                as np
import tensorflow           as tf
import matplotlib.pyplot    as plt

logdir = './logs/'
chkpt = './logs/model.ckpt'
resultsdir = './results/'

class LayerwiseRelevancePropagation:

  def __init__(self):
    self.dataloader = MNISTLoader()
    self.epsilon = 1e-10

    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))

      weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MNIST_CNN')
      self.activations = tf.get_collection('LayerwiseRelevancePropagation')

    self.X = self.activations[0]

    self.act_weights = {}
    for act in self.activations[2:]:
      for wt in weights:
        name = act.name.split('/')[2]
        if name == wt.name.split('/')[2]:
          if name not in self.act_weights:
            self.act_weights[name] = wt

    self.activations = self.activations[:0:-1]
    self.relevances = self.get_relevances()

  def get_relevances(self):
    relevances = [self.activations[0], ]

    for i in range(1, len(self.activations)):
      name = self.activations[i - 1].name.split('/')[2]
      if 'output' in name or 'fc' in name:
        relevances.append(self.backprop_fc(name, self.activations[i], relevances[-1]))
      elif 'flatten' in name:
        relevances.append(self.backprop_flatten(self.activations[i], relevances[-1]))
      elif 'max_pool' in name:
        relevances.append(self.backprop_max_pool2d(self.activations[i], relevances[-1]))
      elif 'conv' in name:
        relevances.append(self.backprop_conv2d(name, self.activations[i], relevances[-1]))
      else:
        raise 'Error parsing layer!'    

    return relevances

  def backprop_fc(self, name, activation, relevance):
    w = self.act_weights[name]
    w_pos = tf.maximum(0.0, w)
    z = tf.matmul(activation, w_pos) + self.epsilon
    s = relevance / z
    c = tf.matmul(s, tf.transpose(w_pos))
    return c * activation

  def backprop_flatten(self, activation, relevance):
    shape = activation.get_shape().as_list()
    shape[0] = -1
    return tf.reshape(relevance, shape)

  def backprop_max_pool2d(self, activation, relevance, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    z = tf.nn.max_pool(activation, ksize, strides, padding='SAME') + self.epsilon
    s = relevance / z
    c = gen_nn_ops.max_pool_grad_v2(activation, z, s, ksize, strides, padding='SAME')
    return c * activation

  def backprop_conv2d(self, name, activation, relevance, strides=[1, 1, 1, 1]):
    w = self.act_weights[name]
    w_pos = tf.maximum(0.0, w)
    z = tf.nn.conv2d(activation, w_pos, strides, padding='SAME') + self.epsilon
    s = relevance / z
    c = tf.nn.conv2d_backprop_input(tf.shape(activation), w_pos, s, strides, padding='SAME')
    return c * activation

  def get_heatmap(self, digit):
    samples = self.dataloader.get_samples(n_samples=1, digit=digit)

    with tf.Session() as sess:    
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))

      heatmap = sess.run(self.relevances[-1], feed_dict={self.X: samples})[0].reshape(28, 28)
      heatmap /= heatmap.max()
    
    return heatmap

  def test(self):
    samples = self.dataloader.get_samples(n_samples=1, digit=np.random.choice(10))
    
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))
      
      R = sess.run(self.relevances, feed_dict={self.X: samples})
      for r in R:
        print(r.sum())

if __name__ == '__main__':
  lrp = LayerwiseRelevancePropagation()
  lrp.test()

  for digit in range(10):
    heatmap = lrp.get_heatmap(digit)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.imshow(heatmap, cmap='Reds', interpolation='bilinear')
    
    fig.savefig('{0}{1}.jpg'.format(resultsdir, digit))  
  