---
tags: autoencoder,encoder,decoder,latent space,covariance matrix,expected value,gaussian distribution,Kullback-Leibler divergence
mathjax: true
---
# Variational Autoencoder (VAE)

{:.caption}
![fully connected layer forward pass](assets/images/deriving_the_KL_divergence_loss_for_vaes.png)
[Deriving the Kullback-Leibler divergence for VAEs](https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes)

<div class="clearfix"></div>

- **p** and **q** : probability distributions.
- **$$\mathcal{N}$$** : a gaussian normal distribution, represented by an [expected value vector](https://en.wikipedia.org/wiki/Expected_value) and a [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix).
- **$$\pmb{I}$$** : a pure diagonal matrix with all diagonal elements set to 1.
- **$$\Sigma$$** : the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of a distribution. In case of the VAE, all covariance matrices are pure diagonal and can therefore be represented by a diagonal vector).
- **$$\sigma^2$$** : a variance value.
- **$$\mu$$** : a mean value.
- **tr$$\{A\}$$** : the [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) of the matrix, which equals the sum over all diagonal elements of the matrix.
- **$$\vert A \vert$$** : the [determinant](https://en.wikipedia.org/wiki/Determinant) of the matrix, which - in case of a pure diagonal matrix - equals the product over all diagonal elements of the matrix.
- **log** : the natural log is a good choice when calculating the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions) of multivariate normal distributions.
