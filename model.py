import tensorflow as tf

class MNIST_CNN:
  
  def __init__(self, name='MNIST'):
    self.name = name
  
  def __call__(self, images, reuse=False):
    with tf.variable_scope(self.name) as scope:
      
      if reuse:
        scope.reuse_variables()
      
      with tf.variable_scope('input_layer'):
        X = tf.reshape(images, (-1, 28, 28, 1))
        
      with tf.variable_scope('block1'):
        conv1 = tf.layers.conv2d(inputs=X, 
                                 filters=32, 
                                 kernel_size=(3, 3), 
                                 padding='SAME', 
                                 activation=tf.nn.relu, 
                                 use_bias=False)
                                 
        pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                        pool_size=(2, 2), 
                                        padding='SAME', 
                                        strides=2)
        
      with tf.variable_scope('block2'):
        conv2 = tf.layers.conv2d(inputs=pool1, 
                                 filters=64, 
                                 kernel_size=(3, 3), 
                                 padding='SAME', 
                                 activation=tf.nn.relu, 
                                 use_bias=False)
                                 
        pool2 = tf.layers.max_pooling2d(inputs=conv2, 
                                        pool_size=(2, 2), 
                                        padding='SAME', 
                                        strides=2)
                                        
      with tf.variable_scope('block3'):
        conv3 = tf.layers.conv2d(inputs=pool2, 
                                 filters=128, 
                                 kernel_size=(3, 3), 
                                 padding='SAME', 
                                 activation=tf.nn.relu, 
                                 use_bias=False)
                                 
        pool3 = tf.layers.max_pooling2d(inputs=conv3, 
                                        pool_size=(2, 2), 
                                        padding='SAME', 
                                        strides=2)
        
      with tf.variable_scope('dense1'):
        flat = tf.reshape(pool3, (-1, 4 * 4 * 128))
        
        dense = tf.layers.dense(inputs=flat, units=512, 
                                 activation=tf.nn.relu, 
                                 use_bias=False)
      
      with tf.variable_scope('dense2'):
        logits = tf.layers.dense(inputs=dense, units=10, 
                                 use_bias=False)
        preds = tf.nn.softmax(logits)
        
      params = [X, conv1, pool1, conv2, pool2, conv3, pool3, flat, dense, preds]
      return params, logits
    
    @property
    def vars(self):
      return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
      
      
      
