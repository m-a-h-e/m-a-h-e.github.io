---
tags: python,numpy,neural-network,successful learning,regularization,dataset,activation function,loss function,learning rate,weight initialization
mathjax: true
---
# Successful Neural Network Learning Notes

## Regularization

- Batch Normalization to move data distribution to the optimal working range
- Weight Decay to keep the weights small which reduces the exploding gradient problem
- Dropout helps to harden the network against distortion
- Reduction of the neural network size (model complexity / number of network parameters)
- Gradient clipping to prevent exploding gradients
- Add data or gradient noise to get better stochastic behavior and improve generalization
- Minimum learning data batch size to stabalize the gradient direction and value

## Dataset

- i.i.d. dataset = independent (diverse data) and identically distributed (the same amount of data items per class)
- Data normalization ... be careful with class target data normalization
- A lot of data ... the larger the network, the more data is needed for training
- Separate training, validation and test datasets are the base for an unbiased network performance (accuracy) evaluation
- A well chosen training data batch size and a random training data selection are important for a stable learning process

## Activation Function

- For the last layer, check that the output range covers the (normalized) target data range
- Check the gradient behavior of the activation function
- The gradient should not get too small and not too big
- Check if the activation function introduces an offset (in case y is not 0 if x is 0)

## Loss Function

- The choice of a suitable loss function is very important and depends on the problem to be solved
- Root-Mean-Square (RMS) Loss for most regression problems
- Softmax + Cross Entropy Loss for multi-class problems where only one class of all is the correct one
- Sigmoid + Binary Cross Entropy Loss for Yes/No problems or if multiple classes are correct for a given input value
- Kullback-Leiber Loss to calculate the distance between two multivariate normal distributions

## Learning Rate

- A large learning rate makes the learning process instable
- A too small learning rate makes learning unnecessary slow
- Find the tradeoff between stability and learning speed
- Select an optimization algorithm (like Adam) which dynamically adjusts the learning rate and introduces momentum

## Weight Initialization

- A proper weight initialization is the base for a successful learning process
- The layer weights (parameters) shall be initialized in a way that the passed through data and the
  back-passed gradients keep their mean value through the network and do not explode or vanish
- Initial neural network weights shall be random values which allows each neuron to become a specialist for a particular pattern ... the diversity of the initial weights is the base for learning a complex function. If all weights of a layer would be initialized with the same weight value they would all behave equal and output the same value which prevents learning complex functions

