import sys

from keras                        import backend as K
from tensorflow.python.ops        import gen_nn_ops
from keras.applications.vgg16     import VGG16
from keras.applications.vgg19     import VGG19
from utils                        import (get_model_params, 
                                          get_gammas, 
                                          get_heatmaps, 
                                          load_images,
                                          predict_labels, 
                                          visualize_heatmap)

images_dir = './images/'
results_dir = './results/'

class LayerwiseRelevancePropagation:

  def __init__(self, model_name='vgg16', alpha=2, epsilon=1e-7):
    model_name = model_name.lower()
    if model_name == 'vgg16':
      model_type = VGG16
    elif model_name == 'vgg19':
      model_type = VGG19
    else:
      raise 'Model name not one of VGG16 or VGG19'
      sys.exit()
    self.model = model_type(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    self.alpha = alpha
    self.beta = 1 - alpha
    self.epsilon = epsilon

    self.names, self.activations, self.weights = get_model_params(self.model)
    self.num_layers = len(self.names)

    self.relevance = self.compute_relevances()
    self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])

  def compute_relevances(self):
    r = self.model.output
    for i in range(self.num_layers-2, -1, -1):
      if 'fc' in self.names[i + 1]:
        r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
      elif 'flatten' in self.names[i + 1]:
        r = self.backprop_flatten(self.activations[i], r)
      elif 'pool' in self.names[i + 1]:
        r = self.backprop_max_pool2d(self.activations[i], r)
      elif 'conv' in self.names[i + 1]:
        r = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
      else:
        raise 'Layer not recognized!'
        sys.exit()
    return r

  def backprop_fc(self, w, b, a, r):
    w_p = K.maximum(w, 0.)
    b_p = K.maximum(b, 0.)
    z_p = K.dot(a, w_p) + b_p + self.epsilon
    s_p = r / z_p
    c_p = K.dot(s_p, K.transpose(w_p))
    
    w_n = K.minimum(w, 0.)
    b_n = K.minimum(b, 0.)
    z_n = K.dot(a, w_n) + b_n - self.epsilon
    s_n = r / z_n
    c_n = K.dot(s_n, K.transpose(w_n))

    return a * (self.alpha * c_p + self.beta * c_n)

  def backprop_flatten(self, a, r):
    shape = a.get_shape().as_list()
    shape[0] = -1
    return K.reshape(r, shape)

  def backprop_max_pool2d(self, a, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    z = K.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], padding='valid', pool_mode='max')

    z_p = K.maximum(z, 0.) + self.epsilon
    s_p = r / z_p
    c_p = gen_nn_ops.max_pool_grad_v2(a, z_p, s_p, ksize, strides, padding='VALID')

    z_n = K.minimum(z, 0.) - self.epsilon
    s_n = r / z_n
    c_n = gen_nn_ops.max_pool_grad_v2(a, z_n, s_n, ksize, strides, padding='VALID')

    return a * (self.alpha * c_p + self.beta * c_n)

  def backprop_conv2d(self, w, b, a, r, strides=(1, 1, 1, 1)):
    w_p = K.maximum(w, 0.)
    b_p = K.maximum(b, 0.)
    z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p + self.epsilon
    s_p = r / z_p
    c_p = K.tf.nn.conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding='SAME')

    w_n = K.minimum(w, 0.)
    b_n = K.minimum(b, 0.)
    z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding='same') + b_n - self.epsilon
    s_n = r / z_n
    c_n = K.tf.nn.conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding='SAME')

    return a * (self.alpha * c_p + self.beta * c_n)

  def predict_labels(self, images):
    return predict_labels(self.model, images)

  def run_lrp(self, images):
    print("Running LRP on {0} images...".format(len(images)))
    return self.lrp_runner([images, ])[0]

  def compute_heatmaps(self, images, g=0.2, cmap_type='rainbow', **kwargs):
    lrps = self.run_lrp(images)
    print("LRP run successfully...")
    gammas = get_gammas(lrps, g=g, **kwargs)
    print("Gamma Correction completed...")
    heatmaps = get_heatmaps(gammas, cmap_type=cmap_type, **kwargs)
    return heatmaps

if __name__ == '__main__':
  image_names = [
    'banana.jpg'
  ]

  image_names += sys.argv[1:]

  image_paths = [images_dir + name for name in image_names]
  image_names = [name.split('.')[0] for name in image_names]

  num_images = len(image_names)
  raw_images, processed_images = load_images(image_paths)
  print("Images loaded...")

  lrp = LayerwiseRelevancePropagation()
  labels = lrp.predict_labels(processed_images)
  print("Labels predicted...")
  heatmaps = lrp.compute_heatmaps(processed_images)
  print("Heatmaps generated...")

  for img, hmap, label, name in zip(raw_images, heatmaps, labels, image_names):
    visualize_heatmap(img, hmap, label, results_dir + name + '.jpg')
