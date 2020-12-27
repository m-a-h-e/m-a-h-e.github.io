---
tags: python,numpy,neural-network,machine learning,supervised learning,unsupervised learning,reinforcement learning,regression,classification,clustering,dimensionality reduction
---
# Machine Learning Fields

>Machine Learning algorithms are usualy classified into 3 to 4 groups.

The first group contains algorithms which learn the relation between given input-output data pairs.
Because the input as well as the output (target) values are given, this group is called Supervised Learning.

The next group of algorithms are used to find the underlying structure of data without telling the algorithm what it has to find. For this reason this group is called Unsupervised Learning.

The third group is somehow a mixture of the properties of the first two and is therefore called Semi-Supervised Learning.
Another main group of machine learning algorithms is called Reinforcement Learning.

These algorithms solve problems by exploring an environment and learning from - sometimes very sparse - returned reward which can be either positive or negative.

## Supervised Learning

>Building a model from labeled data or input-output data pairs.

#### Regression

- Neural Networks
  - [Linear Regression](single_layer_regression.md)
  - [Logistic Regression](single_layer_regression.md)
  - [Non-Linear Regression](non_linear_regression.md)

#### Classification

- Neural Networks
  - [Logistic Regression + Softmax](classification_basics.md)
  - [Convolutional Neural Networks](convolutional_networks.md)
- K Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Trees
- Random Forests
- Naive Bayes

#### Time Series Prediction

- Neural Networks
  - [Recurrent Neural Networks (RNN)](recurrent_networks.md)
  - [Long Short Term Memory (LSTM)](long_short_term_memory.md)
  - Gated Recurrent Unit (GRU)

## Unsupervised Learning

>Learning the underlying structure of data without given labels.

#### Clustering

- K-Means

#### Dimensionality Reduction

- Principal Component Analysis (PCA)

## Semi-Supervised Learning

>Learning the underlying principal feature distribution and generating new but similar data.

#### Generative Models

- [Autoencoders](autoencoder.md) / [Variational Autoencoders (VAEs)](variational_autoencoder.md)
- [Generative Adversarial Networks (GANs)](generative_networks.md)

## Reinforcement Learning

>Solving tasks by exploring an environment and evaluating the returned reward.

#### Explorative Behavior Optimization

- [Temporal Difference (TD) Learning](temporal_difference.md)
- [Deep Q Learning](deep_q_networks.md)
- [Monte Carlo RL Methods](monte_carlo_methods.md)
- [Monte Carlo Tree Search Algorithms](monte_carlo_tree_search.md)
- [Policy Gradient Algorithms](policy_gradient.md)
- [Cross Entropy RL Method](cross_entropy_method.md)

