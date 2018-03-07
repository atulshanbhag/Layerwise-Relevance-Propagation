import tensorflow as tf

class MNIST_CNN:
  
  def __init__(self, name='MNIST'):
    self.name = name
  
  def __call__(self, images, reuse=False):
    with tf.variable_scope(self.name):
      
      if reuse:
        scope.reuse_variables()
      
      params = []

      with tf.variable_scope('input'):
        images = tf.reshape(images, [-1, 28, 28, 1], name='input')
        params += [images, ]

      with tf.variable_scope('conv1'):
        w_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 32], dtype=tf.float32, stddev=0.1), name='w_conv1')
        b_conv1 = tf.Variable(tf.constant(0.0, shape=[32, ], dtype=tf.float32), name='b_conv1')
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(images, w_conv1, [1, 1, 1, 1], padding='SAME'), b_conv1), name='conv1')
        params += [w_conv1, b_conv1]

      with tf.variable_scope('conv2'):
        w_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], dtype=tf.float32, stddev=0.1), name='w_conv2')
        b_conv2 = tf.Variable(tf.constant(0.0, shape=[64, ], dtype=tf.float32), name='b_conv2')
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_conv2, [1, 1, 1, 1], padding='SAME'), b_conv2), name='conv2')
        params += [w_conv2, b_conv2]

      with tf.variable_scope('max_pool1'):
        max_pool1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='max_pool1')
        params += [max_pool1, ]

      with tf.variable_scope('flatten'):
        flatten = tf.contrib.layers.flatten(max_pool1)

      with tf.variable_scope('fc1'):
        n_in = int(flatten.get_shape()[1])
        w_fc1 = tf.Variable(tf.truncated_normal(shape=[n_in, 512], dtype=tf.float32, stddev=0.1), name='w_fc1')
        b_fc1 = tf.Variable(tf.constant(0.0, shape=[512, ], dtype=tf.float32), name='b_fc1')
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, w_fc1), b_fc1), name='fc1')
        params += [w_fc1, b_fc1]

      with tf.variable_scope('dropout1'):
        dropout1 = tf.nn.dropout(fc1, keep_prob=0.5, name='dropout1')

      with tf.variable_scope('output'):
        w_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 10], dtype=tf.float32, stddev=0.1), name='w_fc2')
        b_fc2 = tf.Variable(tf.constant(0.0, shape=[10, ], dtype=tf.float32), name='b_fc2')
        logits = tf.nn.bias_add(tf.matmul(dropout1, w_fc2), b_fc2, name='logits')
        preds = tf.nn.softmax(logits, name='output')
        params += [preds, ]

      return params, logits
    
  @property
  def params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
      
      
      
