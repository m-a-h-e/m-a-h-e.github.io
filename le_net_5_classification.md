---
tags: python,numpy,neural-network,classification,LeNet-5,MNIST
mathjax: true
---
# LeNet-5 Classification

{:.caption .img}
[![Deep Learning, ConvNets, and Self-Supervised Learning](https://img.youtube.com/vi/SGSOCuByo24/0.jpg)](https://www.youtube.com/watch?v=SGSOCuByo24)
[Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) - Deep Learning, ConvNets, and Self-Supervised Learning

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Pad2D(shape_in=(28, 28, 1), pad_axis0=2, pad_axis1=2),
  npnn.Conv2D(shape_in=(32, 32, 1), shape_out=(28, 28, 6), kernel_size=5, stride=1),
  npnn.Tanh(28 * 28 * 6),

  npnn.MaxPool(shape_in=(28, 28, 6), shape_out=(14, 14, 6), kernel_size=2),
  npnn.Conv2D(shape_in=(14, 14, 6), shape_out=(10, 10, 16), kernel_size=5, stride=1),
  npnn.Tanh(10 * 10 * 16),

  npnn.MaxPool(shape_in=(10, 10, 16), shape_out=(5, 5, 16), kernel_size=2),
  npnn.Conv2D(shape_in=(5, 5, 16), shape_out=(1, 1, 120), kernel_size=5, stride=1),
  npnn.Tanh(1 * 1 * 120),

  npnn.Dense(120, 84),
  npnn.Tanh(84),
  npnn.Dense(84, 10),
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
<video controls poster="assets/videos/le_net_5_mnist.png">
  <source src="assets/videos/le_net_5_mnist.webm" type="video/webm">
  <source src="assets/videos/le_net_5_mnist.ogv" type="video/ogg">
  <source src="assets/videos/le_net_5_mnist.mp4" type="video/mp4">
</video>
<p>MNIST Handwritten Digits Classification using Yann LeCun's LeNet-5</p>
</div>

{:.w90}
<div class="video">
<video controls poster="assets/videos/le_net_5_mnist_2.png">
  <source src="assets/videos/le_net_5_mnist_2.webm" type="video/webm">
  <source src="assets/videos/le_net_5_mnist_2.ogv" type="video/ogg">
  <source src="assets/videos/le_net_5_mnist_2.mp4" type="video/mp4">
</video>
<p>MNIST Handwritten Digits Classification using Yann LeCun's LeNet-5<br>
plot of network validation batch data target values (green) and 
predicted network output values (orange)</p>
</div>
