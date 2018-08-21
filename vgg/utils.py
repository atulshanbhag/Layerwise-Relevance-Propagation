import numpy              as np
import matplotlib.pyplot  as plt

from matplotlib.cm                import get_cmap
from keras.applications.vgg16     import (preprocess_input, 
                                          decode_predictions)
from keras.preprocessing.image    import img_to_array, load_img

EPS = 1e-7

def get_model_params(model):
  names, activations, weights = [], [], []
  for layer in model.layers:
    name = layer.name if layer.name != 'predictions' else 'fc_out'
    names.append(name)
    activations.append(layer.output)
    weights.append(layer.get_weights())
  return names, activations, weights

def load_images(image_paths, target_size=(224, 224)):
  raw_images, processed_images = [], []
  for path in image_paths:
    image = load_img(path, target_size=target_size)
    raw_images.append(image)
    image = img_to_array(image)
    image = preprocess_input(image)
    processed_images.append(image)
  return raw_images, np.array(processed_images)

def predict_labels(model, images):
  preds = model.predict(images)
  decoded_preds = decode_predictions(preds)
  labels = [p[0] for p in decoded_preds]
  return labels

def gamma_correction(image, gamma=0.4, minamp=0, maxamp=None):
  c_image = np.zeros_like(image)
  image -= minamp
  if maxamp is None:
    maxamp = np.abs(image).max() + EPS
  image /= maxamp
  pos_mask = (image > 0)
  neg_mask = (image < 0)
  c_image[pos_mask] = np.power(image[pos_mask], gamma)
  c_image[neg_mask] = -np.power(-image[neg_mask], gamma)
  c_image = c_image * maxamp + minamp
  return c_image

def project_image(image, output_range=(0, 1), absmax=None, input_is_positive_only=False):
  if absmax is None:
    absmax = np.max(np.abs(image), axis=tuple(range(1, len(image.shape))))
  absmax = np.asarray(absmax)
  mask = (absmax != 0)
  if mask.sum() > 0:
    image[mask] /= absmax[mask]
  if not input_is_positive_only:
    image = (image + 1) / 2
  image = image.clip(0, 1)
  projection = output_range[0] + image * (output_range[1] - output_range[0])
  return projection

def reduce_channels(image, axis=-1, op='sum'):
  if op == 'sum':
    return image.sum(axis=axis)
  elif op == 'mean':
    return image.mean(axis=axis)
  elif op == 'absmax':
    pos_max = image.max(axis=axis)
    neg_max = -((-image).max(axis=axis))
    return np.select([pos_max >= neg_max, pos_max < neg_max], [pos_max, neg_max])

def heatmap(image, cmap_type='rainbow', reduce_op='sum', reduce_axis=-1, **kwargs):
  cmap = get_cmap(cmap_type)
  shape = list(image.shape)
  reduced_image = reduce_channels(image, axis=reduce_axis, op=reduce_op)
  projected_image = project_image(reduced_image, **kwargs)
  heatmap = cmap(projected_image.flatten())[:, :3].T
  heatmap = heatmap.T
  shape[reduce_axis] = 3
  return heatmap.reshape(shape)

def get_gammas(images, g=0.4, **kwargs):
  gammas = [gamma_correction(img, gamma=g, **kwargs) for img in images]
  return gammas

def get_heatmaps(gammas, cmap_type='rainbow', **kwargs):
  heatmaps = [heatmap(g, cmap_type=cmap_type, **kwargs) for g in gammas]
  return heatmaps

def visualize_heatmap(image, heatmap, label, savepath=None):
  fig = plt.figure()
  fig.suptitle(label)
  ax0 = fig.add_subplot(121)
  ax0.axis('off')
  ax0.imshow(image)
  ax1 = fig.add_subplot(122)
  ax1.axis('off')
  ax1.imshow(heatmap, interpolation='bilinear')
  if savepath is not None:
    fig.savefig(savepath)


if __name__ == '__main__':
  pass
