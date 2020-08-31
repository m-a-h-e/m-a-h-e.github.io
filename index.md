---
tags: python,numpy,neural-network,activation-functions,loss-functions,optimizer,optimizer-algorithms,derivatives,convolution,pooling,relu,leakyrelu,softmax
---

# A Machine Learning Compendium

At the time I was studying Microelectronics and Computer Science I had the opportunity to take some fascinating machine learning lectures.
Since that time I have followed this topic up to now where the research in neural networks and machine learning gains a lot of momentum.

{:.caption}
![Machine Learning Word Cloud](assets/images/ml_word_cloud.png)
<div class="clearfix"></div>

My interest in all the things surrounding this field of research and the growing amount of available publications led me to the decision to - once again - dive deeper into it.

To understand all the things down to their details, I decided to implement all components of neural networks including the optimization environment from scratch using Python and the NumPy library.

So I started based on my knowledge about neural networks as they were the days I studied, combining it with the latest research outcomes regarding new activation functions, new optimizer algorithms and new network structures, altogether better suited to solve several problems.

As a positive side effect I got a better understanding of Python, NumPy, and PyTorch later on ...

## What is a Neural Network made of ?

>Connected Layers of [Artificial Neurons](artificial_neuron.md)

These layers of neurons are coupled by weighted connections which are adjusted during the learning process in a way to minimize the prediction error of the network normally using some sort of error gradient backpropagation.

A good way to build neural networks is to define the following basic building blocks which make it easy to assemble neural networks of different structure and arbitrary complexity:

#### **Connect layer** *(to implement sums of weighted connections)*

- [Dense connected layer](dense_connected_layer.md)
- [Convolution layer](convolutional_networks.md)
- UpConvolution layer
- Pooling layer

#### **Function layer** *(to implement neuron activation functions)*

- [Linear](https://github.com/maideas/numpy-neural-network/blob/master/Linear.ipynb)
- [ReLU](https://github.com/maideas/numpy-neural-network/blob/master/ReLU.ipynb)
- [LeakyReLU](https://github.com/maideas/numpy-neural-network/blob/master/LeakyReLU.ipynb)
- [Tanh](https://github.com/maideas/numpy-neural-network/blob/master/Tanh.ipynb)
- [Sigmoid](https://github.com/maideas/numpy-neural-network/blob/master/Sigmoid.ipynb)
- [Softplus](https://github.com/maideas/numpy-neural-network/blob/master/Softplus.ipynb)
- [Softmax](classification_basics.md)

#### **Loss layer** *(to implement network error loss functions)*

- RMS loss *(= L2 Norm loss)*
- L1 Norm loss
- Cross Entropy loss
- Binary Cross Entropy loss
- Kullback-Leibler loss

#### **Complex layer**

- Sequential layer
- Inception layer
- Latent and Sample layer
- LSTM layer
- GRU layer

To adjust the weights *(parameters)* of the network, an optimization algorithm is needed:

#### **Optimizer**

- [An excellent overview by Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/)
- Stochastic gradient descent
- RMSprop
- Adagrad
- Adadelta
- Adam

