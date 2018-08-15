import tensorflow as tf

class MNIST_CNN:
  
  def __init__(self, name='MNIST_CNN'):
    self.name = name

  def convlayer(self, input, shape, name):
    w_conv = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
    b_conv = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w_conv, [1, 1, 1, 1], padding='SAME'), b_conv), name=name)
    return w_conv, b_conv, conv
  
  def fclayer(self, input, shape, name, prop=True):
    w_fc = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1), name='w_{0}'.format(name))
    b_fc = tf.Variable(tf.constant(0.0, shape=shape[-1:], dtype=tf.float32), name='b_{0}'.format(name))
    if prop:
      fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w_fc), b_fc), name=name)
      return w_fc, b_fc, fc
    else:
      return w_fc, b_fc

  def __call__(self, images, reuse=False):
    with tf.variable_scope(self.name):
      
      if reuse:
        scope.reuse_variables()
      
      activations = []

      with tf.variable_scope('input'):
        images = tf.reshape(images, [-1, 28, 28, 1], name='input')
        activations += [images, ]

      with tf.variable_scope('conv1'):
        w_conv1, b_conv1, conv1 = self.convlayer(images, [3, 3, 1, 32], 'conv1')
        activations += [conv1, ]

      with tf.variable_scope('conv2'):
        w_conv2, b_conv2, conv2 = self.convlayer(conv1, [3, 3, 32, 64], 'conv2')
        activations += [conv2, ]

      with tf.variable_scope('max_pool1'):
        max_pool1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool1')
        activations += [max_pool1, ]

      with tf.variable_scope('flatten'):
        flatten = tf.contrib.layers.flatten(max_pool1)
        activations += [flatten, ]

      with tf.variable_scope('fc1'):
        n_in = int(flatten.get_shape()[1])
        w_fc1, b_fc1, fc1 = self.fclayer(flatten, [n_in, 512], 'fc1')
        activations += [fc1, ]

      with tf.variable_scope('dropout2'):
        dropout2 = tf.nn.dropout(fc1, keep_prob=0.5, name='dropout2')

      with tf.variable_scope('output'):
        w_fc2, b_fc2 = self.fclayer(dropout2, [512, 10], 'fc2', prop=False)
        logits = tf.nn.bias_add(tf.matmul(dropout2, w_fc2), b_fc2, name='logits')
        preds = tf.nn.softmax(logits, name='output')
        activations += [preds, ]

      return activations, logits
    
  @property
  def params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    