---
tags: python,numpy,neural-network,artificial neuron,linear regression,logistic regression
mathjax: true
---
# Single Layer Regression

## Linear Regression

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Dense(1, 1),
  npnn.Linear(1)
]

loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=2e-2)
dataset    = npnn_datasets.NoisyLinear()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/linear_regression.png">
  <source src="assets/videos/linear_regression.webm" type="video/webm">
  <source src="assets/videos/linear_regression.ogv" type="video/ogg">
  <source src="assets/videos/linear_regression.mp4" type="video/mp4">
</video>
<p>Linear Regression (Single Linear Neuron)</p>
</div>

## Logistic Regression

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

loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=2e-2)
dataset    = npnn_datasets.XORFunction()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/xor_function_regression.png">
  <source src="assets/videos/xor_function_regression.webm" type="video/webm">
  <source src="assets/videos/xor_function_regression.ogv" type="video/ogg">
  <source src="assets/videos/xor_function_regression.mp4" type="video/mp4">
</video>
<p>XOR Function Regression (Small LeakyReLU-based Network)</p>
</div>

