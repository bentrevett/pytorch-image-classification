# PyTorch Image Classification

This repo contains tutorials covering how to do sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch) 1.4 and [TorchVision](https://github.com/pytorch/vision) 0.5 using Python 3.7.

We'll start by implementing a multilayer perceptron (MLP) and then move on to architectures using convolutional neural networks (CNNs). Specifically, we'll implement [LeNet](), [AlexNet](), [VGG]() and [ResNet]()

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-image-classification/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

The instructions to install PyTorch should also detail how to install TorchVision but can also be installed via:

``` bash
pip install torchvision
```

## Tutorials

* 1 - [Multilayer Perceptron](https://github.com/bentrevett/pytorch-image-classification/blob/master/1%20-%20MLP.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/bentrevett/pytorch-image-classification/blob/master/1%20-%20MLP.ipynb)

    This tutorial provides an introduction to PyTorch and TorchVision. We'll learn how to: load datasets, augment data, define a multilayer perceptron (MLP), train a model, view the outputs of our model, visualize the model's representations, and view the weights of the model. 

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/pytorch/tutorials
- https://github.com/pytorch/examples
- https://colah.github.io/posts/2014-10-Visualizing-MNIST/
- https://distill.pub/2016/misread-tsne/
- https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b