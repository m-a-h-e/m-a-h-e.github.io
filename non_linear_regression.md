---
tags: python,numpy,neural-network,non-linear regression
mathjax: true
---
# Non-Linear Regression

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
    npnn.Dense(1, 10),
    npnn.Tanh(10),
    npnn.Dense(10, 20),
    npnn.Tanh(20),
    npnn.Dense(20, 40),
    npnn.Tanh(40),
    npnn.Dense(40, 80),
    npnn.Tanh(80),
    npnn.Dense(80, 40),
    npnn.Tanh(40),
    npnn.Dense(40, 20),
    npnn.Tanh(20),
    npnn.Dense(20, 10),
    npnn.Tanh(10),
    npnn.Dense(10, 1),
    npnn.Linear(1)
]

loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=5e-4)
dataset    = npnn_datasets.NoisySine()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/non_linear_regression_tanh.png">
  <source src="assets/videos/non_linear_regression_tanh.webm" type="video/webm">
  <source src="assets/videos/non_linear_regression_tanh.ogv" type="video/ogg">
  <source src="assets/videos/non_linear_regression_tanh.mp4" type="video/mp4">
</video>
<p>Non-Linear Regression (Tanh-based Network)</p>
</div>

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Dense(1, 10),
  npnn.LeakyReLU(10),
  npnn.Dense(10, 20),
  npnn.LeakyReLU(20),
  npnn.Dense(20, 40),
  npnn.LeakyReLU(40),
  npnn.Dense(40, 80),
  npnn.LeakyReLU(80),
  npnn.Dense(80, 40),
  npnn.LeakyReLU(40),
  npnn.Dense(40, 20),
  npnn.LeakyReLU(20),
  npnn.Dense(20, 10),
  npnn.LeakyReLU(10),
  npnn.Dense(10, 1),
  npnn.Linear(1)
]

loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=1e-3)
dataset    = npnn_datasets.NoisySine()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/non_linear_regression_leaky_relu.png">
  <source src="assets/videos/non_linear_regression_leaky_relu.webm" type="video/webm">
  <source src="assets/videos/non_linear_regression_leaky_relu.ogv" type="video/ogg">
  <source src="assets/videos/non_linear_regression_leaky_relu.mp4" type="video/mp4">
</video>
<p>Non-Linear Regression (LeakyReLU-based Network)</p>
</div>

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
    npnn.Dense(1, 10),
    npnn.Swish(10),
    npnn.Dense(10, 20),
    npnn.Swish(20),
    npnn.Dense(20, 40),
    npnn.Swish(40),
    npnn.Dense(40, 80),
    npnn.Swish(80),
    npnn.Dense(80, 40),
    npnn.Swish(40),
    npnn.Dense(40, 20),
    npnn.Swish(20),
    npnn.Dense(20, 10),
    npnn.Swish(10),
    npnn.Dense(10, 1),
    npnn.Linear(1)
]

loss_layer = npnn.loss_layer.RMSLoss(1)
optimizer  = npnn.optimizer.Adam(alpha=5e-4)
dataset    = npnn_datasets.NoisySine()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w70}
<div class="video">
<video controls poster="assets/videos/non_linear_regression_swish.png">
  <source src="assets/videos/non_linear_regression_swish.webm" type="video/webm">
  <source src="assets/videos/non_linear_regression_swish.ogv" type="video/ogg">
  <source src="assets/videos/non_linear_regression_swish.mp4" type="video/mp4">
</video>
<p>Non-Linear Regression (Swish-based Network)</p>
</div>

