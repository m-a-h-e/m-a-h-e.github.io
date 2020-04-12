---
tags: python,numpy,neural-network,successful learning,regularization,weight decay,ridge,lasso,gradient clipping,dataset,normalization,activation function,loss function,learning rate,stable learning,momentum,weight initialization
mathjax: true
---
# Successful Neural Network Learning Notes

## Regularization

- Batch Normalization to move data distribution to the optimal working range
- Weight decay ("Ridge Regression" = L2 Norm weight decay, "Lasso Regression" = L1 Norm weight decay) to keep the weights small which helps to tackle overfitting if the number of data samples is small compared to the network complexity
- Dropout helps against overfitting by randomly blanking a percentage of neurons during training which reduces functional dependencies between neurons. This gain in neuron independency tackles model overfitting which helps to close the gap between training and validation accuracy
- Reducing the model complexity (the number of network parameters) if there is only a small number of data available for training
- Gradient clipping to prevent exploding gradients
- Add data noise or gradient noise to get better stochastic behavior, improve generalization and possibly escape local optimization minima
- A minimum learning data batch size to stabalize the gradient direction and value

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
- Divide the accumulated layer weight gradients by the training batch size (which gives its mean) to avoid large gradient values and make the learning rate independent of the training batch size

## Weight Initialization

- A proper weight initialization is the base for a successful learning process
- The layer weights (parameters) shall be initialized in a way that the passed through data and the
  back-passed gradients keep their mean value through the network and do not explode or vanish
- Neural network weights shall be initialized using random values which allows each neuron to become a specialist for a particular pattern. The diversity of the initial weights is the base for learning a complex function. If all weights of a neural network layer would be initialized using the same weight value they would all behave equal and output the same value which prevents learning a complex function

## Model Complexity

- In case a model has an insufficient complexity (too few parameters to approximate the desired function) the optimizer will not be able to reduce the loss or increase the model accuracy above a certain level, because the model can only represent a low dimensional approximation of the target function.
- In case the model complexity is higher than needed, the model tends to overfit which will result in poor model generalization and can be detected by an unusual difference between (a good) training data accuracy and (a bad) validation data accuracy. To tackle the overfitting problem, several regularization methods can be used.

