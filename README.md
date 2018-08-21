# Layerwise-Relevance-Propagation
Implementation of Layerwise Relevance Propagation for heatmapping "deep" layers, using Tensorflow and Keras.

## Results

### MNIST
<p float="left">
  <img src="mnist/results/0.jpg" width="150" />
  <img src="mnist/results/1.jpg" width="150" /> 
  <img src="mnist/results/2.jpg" width="150" />
  <img src="mnist/results/3.jpg" width="150" />
  <img src="mnist/results/4.jpg" width="150" />
</p>
<p float="left">
  <img src="mnist/results/5.jpg" width="150" />
  <img src="mnist/results/6.jpg" width="150" /> 
  <img src="mnist/results/7.jpg" width="150" />
  <img src="mnist/results/8.jpg" width="150" />
  <img src="mnist/results/9.jpg" width="150" />
</p>

### VGG
<p float="center">
  <img src="vgg/results/apple.jpg" width="400" />
  <img src="vgg/results/balloon.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/banana.jpg" width="400" />
  <img src="vgg/results/bball.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/bison.jpg" width="400" />
  <img src="vgg/results/boxer.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/bullfrog.jpg" width="400" />
  <img src="vgg/results/chihuahua.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/chimp.jpg" width="400" />
  <img src="vgg/results/coffee.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/eagle.jpg" width="400" />
  <img src="vgg/results/flamingo.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/gyromitra.jpg" width="400" />
  <img src="vgg/results/jellyfish.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/orange.jpg" width="400" />
  <img src="vgg/results/ostrich.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/pizza.jpg" width="400" />
  <img src="vgg/results/rifle.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/scorpion.jpg" width="400" />
  <img src="vgg/results/snorkel.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/stingray.jpg" width="400" />
  <img src="vgg/results/teapot.jpg" width="400" />
</p>
<p float="center">
  <img src="vgg/results/volcano.jpg" width="400" />
  <img src="vgg/results/watersnake.jpg" width="400" />
</p>

## Instructions

### MNIST
* Run `train.py` to train model. 
* Weights will be saved in `logs/`. 
* Run `lrp.py` for Layerwise Relevance Propagation.

NOTE: If using Tensorflow version < `1.5.0`, you need to change 
`tf.nn.softmax_cross_entropy_with_logits_v2` to `tf.nn.softmax_cross_entropy_with_logits`.

### VGG
* Feed a list of images to run Layerwise Relevance Propagation on all images.
* All results will be saved in `results/`.
* Run `lrp.py <image_1> <image_2> ... <image_n>`.


## Reference
* [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
* [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://arxiv.org/abs/1512.02479)
* [Understanding Neural Networks with Layerwise Relevance Propagation and Deep Taylor Series](http://danshiebler.com/2017-04-16-deep-taylor-lrp/)
* [A Quick Introduction to Deep Taylor Decomposition](http://heatmapping.org/deeptaylor/)
* [Tutorial: Implementing Layer-Wise Relevance Propagation](http://www.heatmapping.org/tutorial/)
