---
tags: python,numpy,neural-network,successful learning
mathjax: true
---
# Important Notes for Successful Machine Learning

## Regularization

- Batch Normalization to move data distribution to the optimal working range
- Weight Decay to keep the weights small which reduces the exploding gradient problem
- Dropout helps to harden the network against distortion
- Reduction of the network size (model complexity)
- Gradient clipping to prevent exploding gradients
- Add data or gradient noise to get better stochastic behavior and improve generalization
- Minimum learning data batch size to stabalize the gradient direction and value

## Data

- i.i.d. = independent and identically distributed data
- normalization of the data ... but be careful with target (class) data normalization
- a lot of data ... the larger the network, the more data are needed for training
- separate training, validation and test datasets
- the training data batch size and a random training data selection are important

## Activation Function

- for the last layer, check that the output range covers the (normalized) target data range
- check the gradient behavior of the activation function
- the gradient should not get too small and not too big
- check if the activation function introduces an offset (in case y is not 0 if x is 0)

## Loss Function

- the choice of a suitable loss function is very important and depends on the problem to be solved
- Root-Mean-Square (RMS) Loss for regression problems
- Softmax + Cross Entropy Loss for multi-class problems where only one class of all is the correct one
- Sigmoid + Binary Cross Entropy Loss for Yes/No problems or if multiple classes are correct for a given input value
- Kullback-Leiber Loss to calculate the distance between two multivariate normal distributions

## Learning Rate

- a large learning rate makes the learning process instable
- a too small learning rate makes learning slow
- find the tradeoff between stability and learning speed
- select an optimization algorithm (like Adam) which dynamically adjusts the learning rate

## Weight Initialization

- a proper weight initialization is the base for a good learning process
- the layer weights (parameters) shall be initialized in a way that the forwarded data and the
  back-passed gradients keep their mean value through the network and do not explode or vanish

