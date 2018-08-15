# Layerwise-Relevance-Propagation
Implementation of Layerwise Relevance Propagation for heatmapping "deep" layers, using Tensorflow.

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

## Instructions

* Run `train.py` to train model. 
* Weights will be saved in `logs/`. 
* Run `lrp.py` for Layerwise Relevance Propagation.

NOTE: If using Tensorflow version < `1.5.0`, you need to change 
`tf.nn.softmax_cross_entropy_with_logits_v2` to `tf.nn.softmax_cross_entropy_with_logits`.


## Reference
* [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
* [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://arxiv.org/abs/1512.02479)
* [Understanding Neural Networks with Layerwise Relevance Propagation and Deep Taylor Series](http://danshiebler.com/2017-04-16-deep-taylor-lrp/)
* [A Quick Introduction to Deep Taylor Decomposition](http://heatmapping.org/deeptaylor/)
* [Tutorial: Implementing Layer-Wise Relevance Propagation](http://www.heatmapping.org/tutorial/)
