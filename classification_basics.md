---
tags: python,numpy,neural-network,classification,classes,sigmoid,softmax,XOR Classification
mathjax: true
---
# Classification Basics

## The Softmax Function and its Derivative

>We define n as the softmax output vector index and k as the softmax input vector index. The sum symbol in the formulas below always means "sum over all k" (sum over all input indices).

The softmax function is defined as

$$f(x)[n] = y[n] = \frac{e^{x[n]}}{\sum_{}^{}e^{x[k]}}$$

The derivative of softmax(x)[n] with respect to x[k] has to be divided into two cases

The case in which n equals k

$$f'(x)[n] = y'[n] = \frac{ e^{x[n]} { \sum_{}^{} e^{x[k]} } - e^{x[n]} e^{x[k]}}{ (\sum_{}^{} e^{x[k]})^2 }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * 1 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } (1 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} })$$

$$= y[n] (1 - y[k])$$

The case in which n and k are not equal

$$f'(x)[n] = y'[n] = \frac{ 0 * { \sum_{}^{} e^{x[k]} } - e^{x[n]} e^{x[k]}}{ (\sum_{}^{} e^{x[k]})^2 }$$

$$= 0 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } * 0  - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } \frac{ e^{x[k]} }{ \sum_{}^{} e^{x[k]} }$$

$$= \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} } (0 - \frac{ e^{x[n]} }{ \sum_{}^{} e^{x[k]} })$$

$$= y[n] (0 - y[k])$$

## XOR Classification

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Dense(2, 4),
  npnn.LeakyReLU(4),
  npnn.Dense(4, 4),
  npnn.LeakyReLU(4),
  npnn.Dense(4, 1),
  npnn.Sigmoid(1)
]

loss_layer = npnn.loss_layer.BinaryCrossEntropyLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=5e-3)
dataset    = npnn_datasets.XORBinaryClassifier()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/xor_binary_classifier.png">
  <source src="assets/videos/xor_binary_classifier.webm" type="video/webm">
  <source src="assets/videos/xor_binary_classifier.ogv" type="video/ogg">
  <source src="assets/videos/xor_binary_classifier.mp4" type="video/mp4">
</video>
<p>XOR Classification using Sigmoid + Binary-Cross-Entropy-Loss</p>
</div>

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Dense(2, 4),
  npnn.LeakyReLU(4),
  npnn.Dense(4, 4),
  npnn.LeakyReLU(4),
  npnn.Dense(4, 2),
  npnn.Softmax(2)
]

loss_layer = npnn.loss_layer.CrossEntropyLoss(2)
optimizer  = npnn.optimizer.Adam(alpha=1e-2)
dataset    = npnn_datasets.XORTwoClasses()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/xor_two_classes.png">
  <source src="assets/videos/xor_two_classes.webm" type="video/webm">
  <source src="assets/videos/xor_two_classes.ogv" type="video/ogg">
  <source src="assets/videos/xor_two_classes.mp4" type="video/mp4">
</video>
<p>XOR Classification using Softmax + Cross-Entropy-Loss</p>
</div>

