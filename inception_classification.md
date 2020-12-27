---
tags: python,numpy,neural-network,classification,inception module,MNIST
mathjax: true
---
# Inception Module Classification

{:.caption .img}
[![Inception Network](https://img.youtube.com/vi/KfV8CJh7hE0/0.jpg)](https://www.youtube.com/watch?v=KfV8CJh7hE0)
[Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) - Inception Network, [Deeplearning AI](https://www.deeplearning.ai/)

{:.caption .img}
[![Inception Network Motivation](https://img.youtube.com/vi/C86ZXvgpejM/0.jpg)](https://www.youtube.com/watch?v=C86ZXvgpejM)
[Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) - Inception Network Motivation, [Deeplearning AI](https://www.deeplearning.ai/)

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Inception((28, 28, 1),
        2,
    2,  4,
    2,  4,
        2
  ),
  npnn.MaxPool(
    shape_in=(28, 28, 12), shape_out=(14, 14, 12), 
    kernel_size=2
  ),
  npnn.Inception((14, 14, 12),
        2,
    4,  6,
    4,  6,
        2
  ),
  npnn.MaxPool(
    shape_in=(14, 14, 16), shape_out=(7, 7, 16), 
    kernel_size=2
  ),
  npnn.Inception((7, 7, 16),
        2,
    6,  6,
    6,  6,
        2
  ),
  npnn.Dense((7, 7, 16), 140),
  npnn.LeakyReLU(140),
  npnn.Dense(140, 10),
  npnn.Softmax(10)
]

loss_layer = npnn.loss_layer.CrossEntropyLoss(10)
optimizer  = npnn.optimizer.Adam(alpha=1e-3)
dataset    = npnn_datasets.MNIST_28x28_2560()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w90}
<div class="video">
<video controls poster="assets/videos/inception_mnist.png">
  <source src="assets/videos/inception_mnist.webm" type="video/webm">
  <source src="assets/videos/inception_mnist.ogv" type="video/ogg">
  <source src="assets/videos/inception_mnist.mp4" type="video/mp4">
</video>
<p>MNIST Handwritten Digits Classification using an Inception Module network</p>
</div>

{:.w90}
<div class="video">
<video controls poster="assets/videos/inception_mnist_2.png">
  <source src="assets/videos/inception_mnist_2.webm" type="video/webm">
  <source src="assets/videos/inception_mnist_2.ogv" type="video/ogg">
  <source src="assets/videos/inception_mnist_2.mp4" type="video/mp4">
</video>
<p>MNIST Handwritten Digits Classification using an Inception Module network<br>
plot of network validation batch data target values (green) and 
predicted network output values (orange)</p>
</div>

