---
tags: autoencoder,encoder,decoder,latent space,covariance matrix,expected value,gaussian distribution,Kullback-Leibler divergence
mathjax: true
---
# Variational Autoencoder (VAE)

## What are Autoencoders and VAEs ?

An autoencoder is a structure which encodes a high dimensional input dataset into a low dimensional - bottleneck - latent space representation from which a decoder is used to reconstruct the original input data. The idea behind this structure is to find the main underlying features of a dataset, which represent its characteristics and may show up as latent space features. Both the encoder and decoder can be implemented using neural networks, which learn a latent space representation by optimizing the reconstruction error of data passed through this structure. The post by [Joseph Rocca](https://towardsdatascience.com/@joseph.rocca) perfecly shows, that even a 1-dimensional latent space can represent each complex dataset by heavily overfitting. A classical autoencoder is trained without any mechanism of latent space regularisation and after training the decoder itself can most probably not be used - without unexpected effects - to generate new output data from latent space vectors which the decoder has not seen before during training. Thats the point where VAEs come into play ...

A Variational AutoEncoder is trained like a normal autoencoder to minimize the reconstruction loss, but in addition the latent space representation of the dataset is optimized to form a gaussian normal distribution with zero mean and unit variance. This can be done by using the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions), which calculates the distance between two probability distributions and allows to miminize this distance by gradient descent with respect to the encoder parameters. To get a robust latent space representation, latent space vectors are sampled using a gaussian distribution and optimizing the decoder to correctly map these sampled latent space vectors to the original target vectors. This adds a stochastic element to the learning process and prevents the VAE - together with the latent space regularisation - from overfitting. As a result - after training - the encoder can be used to generate new, better interpolated and more meaningful output data from unseen latent space vectors.

## Deriving the Kullback-Leibler Divergence Loss

{:.caption}
![fully connected layer forward pass](assets/images/deriving_the_KL_divergence_loss_for_vaes.png)
[Deriving the Kullback-Leibler divergence loss](https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes)

<div class="clearfix"></div>

- **x** and **z** : encoder input vector and latent space vector.
- **p** and **q** : probability distributions.
- **$$\mathcal{N}$$** : a gaussian normal distribution, represented by an [expected value vector](https://en.wikipedia.org/wiki/Expected_value) and a [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix).
- **$$\pmb{I}$$** : a diagonal matrix with all diagonal elements set to 1.
- **$$\Sigma$$** : the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of a probability distribution. In case of the VAE, the latent space covariance matrix is defined as a diagonal matrix and can therefore be represented by its diagonal vector. The non-diagonal elements are supposed to be 0, which implies that the latent space features are uncorrelated / independent of each other.
- **$$\sigma^2$$** : a variance vector = vector of squared deviation values.
- **$$\mu$$** : a mean vector.
- **tr$$\{A\}$$** : the [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) of the matrix, which equals the sum over all diagonal elements of the matrix.
- **$$\vert A \vert$$** : the [determinant](https://en.wikipedia.org/wiki/Determinant) of the matrix, which - in case of a diagonal matrix - equals the product over all diagonal elements of the matrix.
- **log** : the natural log is a good choice when calculating the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions) of multivariate normal distributions.

## Related MIT Deep Learning Lecture

{:.caption}
[![Deep Generative Modeling, MIT 6.S191](https://img.youtube.com/vi/rZufA635dq4/0.jpg)](https://www.youtube.com/watch?v=rZufA635dq4)
Deep Generative Modeling, MIT 6.S191 (2020)

<div class="clearfix"></div>

## Some nice Articles Explaining VAEs

- [Understanding Variational Autoencoders (VAEs) - by Joseph Rocca](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- [Variational Autoencoders - by Jeremy Jordan](https://www.jeremyjordan.me/variational-autoencoders/)
- [Variational Autoencoders Explained - by Yoel Zeldes](https://anotherdatum.com/vae.html)
- [Intuitively Understanding Variational Autoencoders - by Irhum Shafkat](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

