---
tags: python,numpy,neural-network,convolution,convolutional networks
mathjax: true
---
# Convolutional Neural Networks (CNNs)

{:.caption .img}
[![Convolutional Neural Networks, MIT 6.S191](https://img.youtube.com/vi/iaSUYvmCekI/0.jpg)](https://www.youtube.com/watch?v=iaSUYvmCekI)
[Alexander Amini](https://www.mit.edu/~amini/) - Convolutional Neural Networks, MIT 6.S191 (2020)

{:.caption .img}
[![Convolutional Neural Networks, Stanford University](https://img.youtube.com/vi/bNb2fEVKeEo/0.jpg)](https://www.youtube.com/watch?v=bNb2fEVKeEo)
[Serena Yeung](https://ai.stanford.edu/~syyeung/) - Convolutional Neural Networks, Stanford University (2017)

{:.caption .img}
[![CNN Architectures, Stanford University](https://img.youtube.com/vi/DAOcjicFr1Y/0.jpg)](https://www.youtube.com/watch?v=DAOcjicFr1Y)
[Serena Yeung](https://ai.stanford.edu/~syyeung/) - CNN Architectures, Stanford University (2017)

## CNN Implementation

```python
import numpy_neural_network as npnn
import npnn_datasets

model = npnn.Sequential()
model.layers = [
  npnn.Pad2D(
    shape_in=(10, 10, 1),
    pad_axis0=2, pad_axis1=2
  ),
  npnn.Conv2D(
    shape_in=(14, 14, 1), shape_out=(10, 10, 6),
    kernel_size=5, stride=1
  ),
  npnn.LeakyReLU(10 * 10 * 6),

  npnn.MaxPool(
    shape_in=(10, 10, 6), shape_out=(5, 5, 6),
    kernel_size=2
  ),
  npnn.Conv2D(
    shape_in=(5, 5, 6), shape_out=(2, 2, 10),
    kernel_size=3, stride=2
  ),
  npnn.LeakyReLU(2 * 2 * 10),

  npnn.MaxPool(
    shape_in=(2, 2, 10), shape_out=(1, 1, 10),
    kernel_size=2
  ),
  npnn.LeakyReLU(1 * 1 * 10),

  npnn.Dense((1, 1, 10), 4),
  npnn.Softmax(4)
]

loss_layer = npnn.loss_layer.CrossEntropyLoss(4)
optimizer  = npnn.optimizer.Adam(alpha=1e-2)
dataset    = npnn_datasets.FourImgClasses()

optimizer.norm  = dataset.norm
optimizer.model = model
optimizer.model.chain = loss_layer
```

{:.w90}
<div class="video">
<video controls poster="assets/videos/four_img_classes.png">
  <source src="assets/videos/four_img_classes.webm" type="video/webm">
  <source src="assets/videos/four_img_classes.ogv" type="video/ogg">
  <source src="assets/videos/four_img_classes.mp4" type="video/mp4">
</video>
<p>Four Different Image Pattern Classification using a CNN</p>
</div>

{:.w90}
<div class="video">
<video controls poster="assets/videos/four_img_classes_2.png">
  <source src="assets/videos/four_img_classes_2.webm" type="video/webm">
  <source src="assets/videos/four_img_classes_2.ogv" type="video/ogg">
  <source src="assets/videos/four_img_classes_2.mp4" type="video/mp4">
</video>
<p>Four Different Image Pattern Classification using a CNN<br>
plot of network validation batch data target values (green) and 
predicted network output values (orange)</p>
</div>
