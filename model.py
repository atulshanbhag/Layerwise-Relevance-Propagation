import tensorflow as tf

# CNN model defined for MNIST
class MNIST_CNN:
  
  def __init__(self, name='MNIST'):
    self.name = name
  
  # Compile model
  def __call__(self, images, reuse=False):
    with tf.variable_scope(self.name) as scope:
      
      # To reuse variables for retraining
      if reuse:
        scope.reuse_variables()
      
      # Input Layer
      with tf.variable_scope('input_layer'):
        X = tf.reshape(images, (-1, 28, 28, 1))
        
      # Layer #1 -> Convolution + ReLU + MaxPooling  
      with tf.variable_scope('layer1'):
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
      
      # Layer #2 -> Convolution + ReLU + MaxPooling  
      with tf.variable_scope('layer2'):
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
      
      # Layer #3 -> Convolution + ReLU + MaxPooling                                  
      with tf.variable_scope('layer3'):
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
      
      # Layer #4 -> Fully Connected + ReLU  
      with tf.variable_scope('layer4'):
        flat = tf.reshape(pool3, (-1, 4 * 4 * 128))
        
        dense = tf.layers.dense(inputs=flat, units=512, 
                                 activation=tf.nn.relu, 
                                 use_bias=False)

        dropout = tf.nn.dropout(dense, keep_prob=0.5)
      
      # Layer #5 -> Logits + SoftMax 
      with tf.variable_scope('layer5'):
        logits = tf.layers.dense(inputs=dropout, units=10, 
                                 use_bias=False)
        preds = tf.nn.softmax(logits)
        
    params = [X, conv1, pool1, conv2, pool2, conv3, pool3, flat, dense, dropout, preds]
    return params, logits
    
  # Trainable params
  @property
  def params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
      
      
      
