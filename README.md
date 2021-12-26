# PyTorch Image Classification

This repo contains tutorials covering image classification using [PyTorch](https://github.com/pytorch/pytorch) 1.7, [torchvision](https://github.com/pytorch/vision) 0.8, [matplotlib](https://matplotlib.org/) 3.3 and [scikit-learn](https://scikit-learn.org/stable/index.html) 0.24, with Python 3.8.

We'll start by implementing a multilayer perceptron (MLP) and then move on to architectures using convolutional neural networks (CNNs). Specifically, we'll implement [LeNet](http://yann.lecun.com/exdb/lenet/), [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGG](https://arxiv.org/abs/1409.1556) and [ResNet](https://arxiv.org/abs/1512.03385).

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-image-classification/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/).

The instructions to install PyTorch should also detail how to install torchvision but can also be installed via:

``` bash
pip install torchvision
```

## Tutorials

* 1 - [Multilayer Perceptron](https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb)

    This tutorial provides an introduction to PyTorch and TorchVision. We'll learn how to: load datasets, augment data, define a multilayer perceptron (MLP), train a model, view the outputs of our model, visualize the model's representations, and view the weights of the model. The experiments will be carried out on the MNIST dataset - a set of 28x28 handwritten grayscale digits.

* 2 - [LeNet](https://github.com/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb)

    In this tutorial we'll implement the classic [LeNet](http://yann.lecun.com/exdb/lenet/) architecture. We'll look into convolutional neural networks and how convolutional layers and subsampling (aka pooling) layers work.

* 3 - [AlexNet](https://github.com/bentrevett/pytorch-image-classification/blob/master/3_alexnet.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/3_alexnet.ipynb)

    In this tutorial we will implement [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), the convolutional neural network architecture that helped start the current interest in deep learning. We will move on to the CIFAR10 dataset - 32x32 color images in ten classes. We show: how to define architectures using `nn.Sequential`, how to initialize the parameters of your neural network, and how to use the learning rate finder to determine a good initial learning rate.

* 4 - [VGG](https://github.com/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb)

    This tutorial will cover implementing the [VGG](https://arxiv.org/abs/1409.1556) model. However, instead of training the model from scratch we will instead load a VGG model pre-trained on the [ImageNet](http://www.image-net.org/challenges/LSVRC/) dataset and show how to perform transfer learning to adapt its weights to the CIFAR10 dataset using a technique called discriminative fine-tuning. We'll also explain how adaptive pooling layers and batch normalization works.

* 5 - [ResNet](https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb)

    In this tutorial we will be implementing the [ResNet](https://arxiv.org/abs/1512.03385) model. We'll show how to load your own dataset, using the [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset as an example, and also how to use learning rate schedulers which dynamically alter the learning rate of your model whilst training. Specifially, we'll use the one cycle policy introduced in [this](https://arxiv.org/abs/1803.09820) paper and is now starting to be commonly used for training computer vision models.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/pytorch/tutorials
- https://github.com/pytorch/examples
- https://colah.github.io/posts/2014-10-Visualizing-MNIST/
- https://distill.pub/2016/misread-tsne/
- https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
- https://github.com/activatedgeek/LeNet-5
- https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
- https://github.com/kuangliu/pytorch-cifar
- https://github.com/akamaster/pytorch_resnet_cifar10
- https://sgugger.github.io/the-1cycle-policy.html
